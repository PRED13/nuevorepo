import os
from django.core.wsgi import get_wsgi_application

# Asegúrate de que 'config.settings' coincida con el nombre de tu carpeta de proyecto
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_wsgi_application()

# Esta línea es un "hack" necesario para que Vercel reconozca la variable 'app'
app = application