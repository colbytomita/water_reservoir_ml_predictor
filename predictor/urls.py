from django.urls import path
from predictor import views

urlpatterns = [
    path("", views.home, name='home'),
    path("charts/", views.charts, name="charts"),
]