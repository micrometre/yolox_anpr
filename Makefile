.PHONY: clean
clean:
	rm -rf images/*
inference_openvino:
	./run_openvino.sh -m models/yolox_s.onnx -i test_images/street_scene.png
infference_trained_model_image_folder:
	python YOLOX/tools/demo.py image  -f  YOLOX/exps/ccpd/yolox_ccpd_nano_test.py  -c YOLOX/YOLOX_outputs/yolox_ccpd_nano_test/latest_ckpt.pth  --path  dataset/CCPD2020/ccpd_green/val  --conf 0.25  --nms 0.45  --tsize 640  --save_result  --device cpu                    
infference_trained_model_video:
	python  YOLOX/tools/demo_video_frames.py -f YOLOX/exps/ccpd/yolox_ccpd_nano_test.py  -c  YOLOX/YOLOX_outputs/yolox_ccpd_nano_test/latest_ckpt.pth     --path ~/Videos/carpark2.mp4    --conf 0.01     --skip-frames 1     --device cpu     --output results/video_output_frames