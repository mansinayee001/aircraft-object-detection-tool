import pytest 
from app import app 
from ultralytics import YOLO
import os


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.fixture(scope='module')
def models():
    model = YOLO('static/results/pretrained_best.pt') # pre-trained best.pt 
    finetuned_model = YOLO('static/results/finetuned_best.pt') # fine-tuned best.pt
    return model, finetuned_model
        

def test_file_upload_test(client):
    """ Tests the file upload by submitting an empty request"""
    response = client.post('/', data={})
    assert b'You need to select 2 images first' in response.data
    assert response.status_code == 200 # checking if the reponse is ok

def test_model_load(models):
    """Ensures the YOLO models are loaded correctly"""
    model, finetuned_model = models
    assert model is not None, "Pre-trained Model failed to load"
    assert finetuned_model is not None, "Fine-tuned Model failed to load"
    

def test_model_inference(models):
    """ Runs inference on a test image to ensure that the functionality works"""
    model, finetuned_model = models
    test_image_path = os.path.join(os.path.dirname(__file__), 'uploads','aircraft_1.jpg')

    # running inference using both models
    results = model(test_image_path)
    results_1 = finetuned_model(test_image_path)

    # Ensuring that a result is returned 
    assert len(results) > 0, "Model did not return processed image"
    assert results[0] is not None, "Model output is empty"

    assert len(results_1) > 0, "Model did not return processed image"
    assert results_1[0] is not None, "Model output is empty"