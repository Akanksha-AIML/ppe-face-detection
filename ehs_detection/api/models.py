from django.db import models

# Create your models here.

from django.db import models

class Detection(models.Model):
    image = models.ImageField(upload_to='detections/')
    assets = models.CharField(max_length=100)
    is_safe = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.assets} - {self.created_at}"

