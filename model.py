import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

class Detection:
    def __init__(self):
        self.dir = dir
        self.model = YOLO('models/detection.pt')
    
    def predict(self, image):
        detection_results = self.model.predict(image, imgsz=1280, conf=0.5, save=False, save_crop=False, save_txt=False)
        return detection_results
    

class Recognition:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("models/recognition/model")
        self.processor = TrOCRProcessor.from_pretrained("models/recognition/processor")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.device = next(self.model.parameters()).device  # get the device of the model

    def predict(self, detection_result, image):
        predictions = []
        for result in detection_result:
            boxes = result.boxes.data.cpu().numpy() # get boxes on cpu in numpy
            for index2, box in enumerate(boxes): # iterate boxes 
                r = box.astype(int)
                class_name = result.names[int(result.boxes.cls[index2])]
                cropped_image = image.crop((r[0], r[1], r[2], r[3]))
                recognition_result = self.predict_box(cropped_image)
                predictions.append({"class_name": class_name, "text_pred": recognition_result})
        return predictions

    def predict_box(self, image):
        self.model.eval()
        with torch.no_grad():
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)  # move the pixel_values tensor to the same device as the model
            outputs = self.model.generate(pixel_values)
            pred_str = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return pred_str
        

