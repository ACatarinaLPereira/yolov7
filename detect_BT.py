#run: python detect_BT.py --source videos_yolo/palace2s.mp4  --weights yolov7.pt --conf 0.1 --img-size 640 --device cpu --names data/drone.names

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.byte_tracker import BYTETracker

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    out, source, weights, view_img, save_txt, imgsz, trace, cfg, names, allign, display_bb = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.cfg, opt.names, opt.allign, opt.display_bb
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Initialize tracker
    tracker = BYTETracker()
    #tracker = Sort()
    #tracker = DeepSort("deepsort_tracker/ckpt.t7")
    print("New tracker")

    img_ant = None
    dx = 0
    dy = 0
    init = False
    frame=0
    id_switches = 0
    track_id_ant = 1   

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if i > 0:
                print('Error in detect_bytetrack_allign')

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                 
                print("detections pre allignment", det[:, :5])
                #convert from inertial to camera: +
                #convert from camera to inercial: -
                if allign: #Compensate for camera motion (express coordinates in the frame of reference of the 1st image)
                    print('inside allign')
                    det[:,0] -= dx
                    det[:,1] -= dy
                    det[:,2] -= dx
                    det[:,3] -= dy
                
                print("detections post allignment", det[:, :5])
                #print("dx, dy", dx, dy)       

                if isinstance(tracker, BYTETracker):     
                    online_targets, not_activated_stracks, lost_stracks = tracker.update(det[:, :5].cpu().detach().numpy(), img_info = [512, 640], img_size = [512, 640])
                elif isinstance(tracker, Sort):
                    online_targets = tracker.update(det[:, :5].cpu().detach().numpy(), img_info = [512, 640], img_size = [512, 640])
                elif isinstance(tracker, DeepSort):
                    #print("detection", det[:, :5].cpu().detach().numpy())
                    online_targets = tracker.update(det[:, :5].cpu().detach().numpy(), img_info = [512, 640], img_size = [512, 640], offset = [dx, dy], ori_img = im0.copy()  )

                #Show references for image stabilization
                if allign and (save_img or view_img):      
                    tlbr = np.array([(100+dx)%600, (200+dy)%600, (100+dx)%600 + 25,  (200+dy)%600 + 25])
                    plot_one_box(tlbr , im0, label='ref', color=[0,100,0], line_thickness=3)
                    tlbr = np.array([(400+dx)%600, (200+dy)%600, (400+dx)%600 + 25,  (200+dy)%600 + 25])
                    plot_one_box(tlbr , im0, label='ref', color=[0,100,0], line_thickness=3)
                    tlbr = np.array([(100+dx)%600, (500+dy)%600, (100+dx)%600 + 25,  (500+dy)%600 + 25])
                    plot_one_box(tlbr , im0, label='ref', color=[0,100,0], line_thickness=3)
                    tlbr = np.array([(400+dx)%600, (500+dy)%600, (400+dx)%600 + 25,  (500+dy)%600 + 25])
                    plot_one_box(tlbr , im0, label='ref', color=[0,100,0], line_thickness=3)

                #print time and detection o txt file               
                if save_txt: 
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(str("time: " + str(frame/25) + str(det[:, :5]) + '\n'))  # label format  'time: %g' % (t.track_id)
        
                for t in online_targets:
                    if isinstance(tracker, BYTETracker):                        
                        tlbr = t.tlbr
                        track_id = t.track_id
                    #elif isinstance(tracker, Sort) or isinstance(tracker, DeepSort):
                        #tlbr =  np.array([t[0], t[1], t[2], t[3]])
                        #track_id = t[4]
                    print("track= ", tlbr, track_id)
                    
                    #Count id switches
                    if track_id != track_id_ant:
                        id_switches +=1
                    track_id_ant = track_id

                    if save_txt: 
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(str(str([dx, dy]) + str(tlbr) + str(track_id) + '\n'))
                    if save_img or view_img:  # Add bbox to image
                        label = 'id: %g' % (track_id)
                        if allign:
                            plot_one_box(tlbr + [dx, dy, dx, dy], im0, label=label, color=[0,0,100], line_thickness=3)
                            #print("plot", tlbr + [dx, dy, dx, dy])
                        else:
                            plot_one_box(tlbr, im0, label=label, color=[0,0,100], line_thickness=3)   

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print('ID switches= ', id_switches)
    print('Done. (%.3fs)' % (time.time() - t0))
    del tracker

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--allign', action='store_true', help='use image allignment to help with tracking')
    parser.add_argument('--display_bb', action='store_true', help='display the bounding boxes returned by the detector')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
