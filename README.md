# Nvprof_Jetson_Nano_Example

### 新增cu_prof_start() and cu_prof_stop() in nvprof.py. 

```
def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.
    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    cu_prof_start() #for nvprof start profiling
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        cu_prof_stop() #for nvprof stop profiling
```

[start profiling function 加於while loop前](https://github.com/F64081169/Nvprof_Jetson_Nano_Example/blob/cd8ce34749ad0286dc5fc76d386ff034f1627f03/trt_yolo.py#L65)

[stop profiling function 加於while loop最後](https://github.com/F64081169/Nvprof_Jetson_Nano_Example/blob/cd8ce34749ad0286dc5fc76d386ff034f1627f03/trt_yolo.py#L87)

- 即可觀察每次inference的執行資訊
### results

![image](https://user-images.githubusercontent.com/62001405/165888935-049607b9-5b3e-4d51-be42-9eec0edd023d.png)
![image](https://user-images.githubusercontent.com/62001405/165888957-72f6d985-31a0-46b6-a232-acb4a06a46bb.png)
![image](https://user-images.githubusercontent.com/62001405/165888981-e282292c-4ac8-4031-b796-a73ba17fae69.png)

### reference
1. [[腾讯机智] tensorflow profiling工具简介——nvprof和nvvp](https://zhuanlan.zhihu.com/p/112857758?fbclid=IwAR3_gtr3t1aD3KLqJxcHVMawz_GE6ke-1F8F1OAyzFZYMacpF1k6n2Tx5tw)
2. [Jetson上的零拷贝](https://zhuanlan.zhihu.com/p/87013236)
3. [NVVP Tutorial Gist](https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086?fbclid=IwAR23KZoL4f5RYhneccVNgWe5NEnXO4L4ooZ22NPQAEMoUCLvYZgqSda6CCM)
4. [Cuda Toolkit Documentation](https://docs.nvidia.com/cuda/profiler-users-guide/index.html?fbclid=IwAR3Zi8LZLOF3ETFxPcBKFREcnSaqC26ZG1uQeYJcoAScpIVCvIwzfkRW0Vc#nvprof-overview)
        
