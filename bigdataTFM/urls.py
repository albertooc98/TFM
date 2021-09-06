from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('bdmed/', include('bdmed.urls')),
    path('admin/', admin.site.urls),
]
