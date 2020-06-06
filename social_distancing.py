import argparse
import pdb

from utils.quant import quant
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.social_distancing import *
from utils.utils import *


def detect(save_img=False):
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(
        device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Quantize model
    # pdb.set_trace()
    if opt.quant:
        with open(opt.hyps) as hyps_file:
            quant_hyp = eval(hyps_file.read())['quant_hyp']
        model = quant(quant_hyp, model)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(
            name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split(
            '.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    if webcam:
        view_img = True
        # set True to speed up constant image size inference
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)
    vid_path, vid_writer = "", None
    bird_vid_path, bird_vid_writer = None, None

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)]
                              for _ in range(len(names))]

    # Init social distancing obj
    sdes = SocialDistancingES(dataset, opt.bird_scale)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()
              ) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Draw ROI
                roi_pts = np.array(dataset.mouse_pts[:4], np.int32)
                cv2.polylines(im0, [roi_pts], True,
                                (0, 255, 255), thickness=4)

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                # Filter boxes with people
                # pdb.set_trace()
                num_persons = 0
                person_boxes = []
                for *xyxy, conf, cls_ in det:
                    if cls_.item() == 0:
                        num_persons += 1
                        # bbox = [xyxy[i] / dataset.w if i % 2 == 0 else xyxy[i] / dataset.h for i in range(len(xyxy))]
                        person_boxes.append(xyxy)

                # Social distancing expert system
                sdes.plot_points_on_bird_eye_view(im0, person_boxes)
                sdes.plot_lines_between_nodes()
                # sdes.calculate_stay_at_home_index()
                sdes.update_stats(num_persons)

            # Display text info on frame
            sdes.draw_info(im0)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                cv2.imshow("Bird's eye", bird_im)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    # pdb.set_trace()
                    if os.path.splitext(save_path)[0] not in vid_path:  # new video
                        # vid_path = save_path
                        vid_path = "_out".join(os.path.splitext(save_path))
                        bird_vid_path = "_bird".join(os.path.splitext(save_path))

                        vid_writer = release_init_writer(vid_path, vid_writer, dataset.fps, opt.fourcc, dataset.w, dataset.h)
                        bird_vid_writer = release_init_writer(bird_vid_path, bird_vid_writer, dataset.fps, opt.fourcc, int((dataset.w*sdes.scale_w)), int((dataset.h*sdes.scale_h)))

                    # pdb.set_trace()
                    vid_writer.write(im0)
                    bird_vid_writer.write(sdes.bird_im)


    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--quant', action='store_true', help='Quantize using quant module')
    parser.add_argument('--hyps', help='Hyps file with quant configuration')
    parser.add_argument('--bird-scale', nargs=2, type=float, default=[0.6, 2], help='Scale for bird\'s eye view video (w,h)')
    opt = parser.parse_args()
    print(opt)

    if opt.quant and opt.hyps is None:
        raise ValueError("You must specify a hyps file in order to run a quantized inference")

    with torch.no_grad():
        detect()
