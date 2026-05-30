import onnxruntime as ort
import numpy as np
import logging
import cv2
import os
import time


def setup_logger():
    os.makedirs("logs",exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/inference.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()


class ONNXInferenceEngine():
    def __init__(self,model_path: str,use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu

        self.options = ort.SessionOptions()
        self.options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


        self.session = ort.InferenceSession(self.model_path,self.options,providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.use_gpu
            else ['CPUExecutionProvider']
        ))
        logger.info(f"Execution providers: {self.session.get_providers()}")
        self._get_model_detail()
        self._warmup()

    def _get_model_detail(self):
        self.input_inputs = self.session.get_inputs()
        self.input_name = self.input_inputs[0].name
        self.input_shape = self.input_inputs[0].shape
        self.input_type = self.input_inputs[0].type

        self.output_outputs = self.session.get_outputs()
        self.output_name = self.output_outputs[0].name
    
    def _warmup(self):

        dummy_shape = [dim if isinstance(dim, int) else 1 for dim in self.input_shape]
        """
        if self.input_shape = [batch_size, C, H , W],
        it loops:
            iteration 1: batch_size
            iteration 2: C
            iteration 3: H
            iteration 4: W
        
        Finally, the dummy_shape becomes same as the self.input_shape. It is a dynamic method
        """

        dummy_input = np.zeros(dummy_shape,dtype=np.float32)
        self.session.run([self.output_name],{self.input_name: dummy_input}) # run([output_name],{ input_name: input_value }, run_options=None)
        logger.info(f"Model warm up sucessfully")
    
    def inference(self,image_path):
        img,resize_img = self.preprocessing(image_path)
        start = time.time()
        outputs = self.session.run([self.output_name],{self.input_name: resize_img})
        logger.info(f"Inference time = {time.time() - start}")
        return self.postprocessing(img,outputs[0])

    def preprocessing(self,image_path):
        img = cv2.imread(image_path)
        resize_img = cv2.resize(img,(640,640))
        resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB) # color first
        resize_img = resize_img.astype(np.float32) / 255.0
        resize_img = resize_img.transpose(2,0,1)
        resize_img = np.expand_dims(resize_img,0)
        # print(img.shape)
        return img,resize_img
    
    def postprocessing(self,img ,model_output):
        
        # print(model_output[0][0])
        x1,y1,x2,y2,conf,cls = model_output[0][0]
        h,w = img.shape[:2]
        img = cv2.resize(img,(640,640))
        return img[max(0,int(y1)):min(h,int(y2)) ,
                   max(0,int(x1)):min(w,int(x2)) ]


    def evaluate(self,image_path):
        img,resize_img = self.preprocessing(image_path)
        times = []

        for _ in range(100):
            start = time.time()
            outputs = self.session.run([self.output_name],{self.input_name: resize_img})
            times.append(time.time() - start)

        logger.info(f"{self.model_path} Average latency: {sum(times)/len(times)}")






if __name__ == "__main__":
    
    model_path = 'infoBox_fp32.onnx'
    model_path_fp16 = 'infoBox_fp16.onnx'
    image_path = './images/IMG_1.jpg'
    # image_path = './images/label4.png'

    engine = ONNXInferenceEngine(model_path)
    engine2 = ONNXInferenceEngine(model_path_fp16)

    engine.evaluate(image_path)
    engine2.evaluate(image_path)

    output = engine.inference(image_path)
    cv2.imwrite("images/cropped_fp32_img.jpg",output)

    output = engine2.inference(image_path)
    cv2.imwrite("images/cropped_fp16_img.jpg",output)






    
