from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from .forms import BreastCancerForm
import os
from django.conf import settings
from keras.models import load_model
import joblib

# Load the scaler and model using absolute paths
scaler_path = os.path.join(settings.BASE_DIR, 'C:\Users\prana\Downloads\Download\Breast cancer Prediction\scaler.pkl')
model_path = os.path.join(settings.BASE_DIR, 'C:\Users\prana\Downloads\Download\Breast cancer Prediction\breast_cancer_model.h5')

# Load scaler
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    raise Exception(f"Scaler file not found at {scaler_path}")

# Load model
try:
    model = load_model(model_path)
except FileNotFoundError:
    raise Exception(f"Model file not found at {model_path}")
except KeyError as e:
    raise Exception(f"Error loading the model: {e}")

def predict(request):
    if request.method == 'POST':
        form = BreastCancerForm(request.POST)
        if form.is_valid():
            data = np.array([[
                form.cleaned_data['mean_radius'],
                form.cleaned_data['mean_texture'],
                form.cleaned_data['mean_perimeter'],
                form.cleaned_data['mean_area'],
                form.cleaned_data['mean_smoothness'],
                form.cleaned_data['mean_compactness'],
                form.cleaned_data['mean_concavity'],
                form.cleaned_data['mean_concave_points'],
                form.cleaned_data['mean_symmetry'],
                form.cleaned_data['mean_fractal_dimension'],
                form.cleaned_data['radius_error'],
                form.cleaned_data['texture_error'],
                form.cleaned_data['perimeter_error'],
                form.cleaned_data['area_error'],
                form.cleaned_data['smoothness_error'],
                form.cleaned_data['compactness_error'],
                form.cleaned_data['concavity_error'],
                form.cleaned_data['concave_points_error'],
                form.cleaned_data['symmetry_error'],
                form.cleaned_data['fractal_dimension_error'],
                form.cleaned_data['worst_radius'],
                form.cleaned_data['worst_texture'],
                form.cleaned_data['worst_perimeter'],
                form.cleaned_data['worst_area'],
                form.cleaned_data['worst_smoothness'],
                form.cleaned_data['worst_compactness'],
                form.cleaned_data['worst_concavity'],
                form.cleaned_data['worst_concave_points'],
                form.cleaned_data['worst_symmetry'],
                form.cleaned_data['worst_fractal_dimension']
            ]])

            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)
            result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'
            return JsonResponse({'result': result})

    return render(request, 'predictor/predict.html', {'form': BreastCancerForm()})
