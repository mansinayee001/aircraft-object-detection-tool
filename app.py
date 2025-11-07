from flask import Flask, render_template, request
from ultralytics import YOLO
import os

class YOLO_Object_Detection:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def process_image(self, image_path):
        """Runs YOLO inference on image and saves to results folder"""
        detection_results = self.model(image_path) # running YOLO inference on image
        result_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_result{os.path.splitext(image_path)[1]}" # appending result to the end of the original path for distinct names
        result_image_path = os.path.join('static/results', result_filename) 
        detection_results[0].save(filename=result_image_path) # saving image to results path
        return result_image_path

class ManageFile:
    @staticmethod
    def save_image(file, folder='uploads'):
        """Saves images to uploads folder"""
        image_path = os.path.join(folder, file.filename)
        file.save(image_path)
        return image_path
    
app = Flask(__name__)

pretrained_model = YOLO_Object_Detection('static/results/pretrained_best.pt') # pre-trained best.pt 
finetuned_model = YOLO_Object_Detection('static/results/finetuned_best.pt') # fine-tuned best.pt    

@app.route("/", methods=["GET", "POST"])
def object_detection():
    """ Performing object detection on both files and ensuring files are selected"""
    result_image_path_1 = None  # stores file path, ensures that value is defined, even before usage
    result_image_path_2 = None 
    error_msg = None

    if request.method == "POST":
       
       if 'file_1' not in request.files or 'file_2' not in request.files or request.files['file_1'].filename == "" or request.files['file_2'].filename == "":
           error_msg = "You need to select 2 images first"
           return render_template("index.html", error=error_msg)
       
       if 'file_1' in request.files: # checks if a file is in requests
           f = request.files['file_1'] # retrieves uploaded file 
           image_path = ManageFile.save_image(f) # saving image to folder
           result_image_path_1 = pretrained_model.process_image(image_path) # perform object detection and saves image to results

       if 'file_2' in request.files:
            f1 = request.files['file_2']
            finetuned_result_image_path = ManageFile.save_image(f1)
            result_image_path_2 = finetuned_model.process_image(finetuned_result_image_path)

    return render_template("index.html", detection_result=result_image_path_1, finetuned_detection_result=result_image_path_2) 

if __name__ == '__main__':
    app.run(debug=True)


