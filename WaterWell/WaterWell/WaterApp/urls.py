from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('Register', views.Register, name="Register"),
	       path('RegisterAction', views.RegisterAction, name="RegisterAction"),
	       path('Visualization', views.Visualization, name="Visualization"),
	       path('Clustering', views.Clustering, name="Clustering"),
	       path('TrainModels', views.TrainModels, name="TrainModels"),
	       path('CurrentWater', views.CurrentWater, name="CurrentWater"),
	       path('CurrentWaterAction', views.CurrentWaterAction, name="CurrentWaterAction"),	
	       path('WaterBearing', views.WaterBearing, name="WaterBearing"),
	       path('WaterBearingAction', views.WaterBearingAction, name="WaterBearingAction"),
	       path('Feedback', views.Feedback, name="Feedback"),
	       path('FeedbackAction', views.FeedbackAction, name="FeedbackAction"),
]