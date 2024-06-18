from django import forms

class BreastCancerForm(forms.Form):
    mean_radius = forms.FloatField(label='Mean Radius', required=True)
    mean_texture = forms.FloatField(label='Mean Texture', required=True)
    mean_perimeter = forms.FloatField(label='Mean Perimeter', required=True)
    mean_area = forms.FloatField(label='Mean Area', required=True)
    mean_smoothness = forms.FloatField(label='Mean Smoothness', required=True)
    mean_compactness = forms.FloatField(label='Mean Compactness', required=True)
    mean_concavity = forms.FloatField(label='Mean Concavity', required=True)
    mean_concave_points = forms.FloatField(label='Mean Concave Points', required=True)
    mean_symmetry = forms.FloatField(label='Mean Symmetry', required=True)
    mean_fractal_dimension = forms.FloatField(label='Mean Fractal Dimension', required=True)
    radius_error = forms.FloatField(label='Radius Error', required=True)
    texture_error = forms.FloatField(label='Texture Error', required=True)
    perimeter_error = forms.FloatField(label='Perimeter Error', required=True)
    area_error = forms.FloatField(label='Area Error', required=True)
    smoothness_error = forms.FloatField(label='Smoothness Error', required=True)
    compactness_error = forms.FloatField(label='Compactness Error', required=True)
    concavity_error = forms.FloatField(label='Concavity Error', required=True)
    concave_points_error = forms.FloatField(label='Concave Points Error', required=True)
    symmetry_error = forms.FloatField(label='Symmetry Error', required=True)
    fractal_dimension_error = forms.FloatField(label='Fractal Dimension Error', required=True)
    worst_radius = forms.FloatField(label='Worst Radius', required=True)
    worst_texture = forms.FloatField(label='Worst Texture', required=True)
    worst_perimeter = forms.FloatField(label='Worst Perimeter', required=True)
    worst_area = forms.FloatField(label='Worst Area', required=True)
    worst_smoothness = forms.FloatField(label='Worst Smoothness', required=True)
    worst_compactness = forms.FloatField(label='Worst Compactness', required=True)
    worst_concavity = forms.FloatField(label='Worst Concavity', required=True)
    worst_concave_points = forms.FloatField(label='Worst Concave Points', required=True)
    worst_symmetry = forms.FloatField(label='Worst Symmetry', required=True)
    worst_fractal_dimension = forms.FloatField(label='Worst Fractal Dimension', required=True)
