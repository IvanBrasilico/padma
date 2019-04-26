from wsgiref.simple_server import make_server
from tg import expose, TGController, AppConfig

from home import *
import numpy as np
import base64
import io
from PIL import Image

global images

class RootController(TGController):
      @expose()
      def index(self, imagem=0, file1=""):
          global images
          #images = imagelist(int(imagem))
          if file1=="":
                output = "<head><meta http-equiv=\"REFRESH\" content=\"0;url=/form\"> </head>"
          else:
                buf = file1.file.read()
                img = Image.open(io.BytesIO(buf))
                img.thumbnail((256,120), Image.ANTIALIAS)
                X = np.asarray(img).reshape(256*120)
                X = X / 255
                images = imagelist2(X)
                output="<html>"
                output+="<img src=\""
                output+=images[0][-15:]
                output+="\"></img><br>"
                output+="<a href=\"buscador?pag=0\">Buscar imagens semelhantes</a><br>"
                output+="<a href=\"form\">Escolher outra imagem</a><br>"
                output+="<html>"
          return output
      @expose()
      def buscador(self, pag=0):
          #images = imagelist()
          global images
          output="<html>"
          output+="<a href=\"form\">Escolher outra imagem</a><br>"
          cont=0
          pag = int(pag)
          for r in range(pag*30, (pag+1)*30):
                cont+=1
                output+="<img src=\""
                output+=images[r][-15:]
                output+="\"></img>&nbsp;"
                if cont % 5 == 0:
                      output+="<br>"
          if pag > 0:
              output+="<a href=\"buscador?pag="+str(pag-1)
              output+="\">Pagina anterior</a>"
          output+="<a href=\"buscador?pag="+str(pag+1)
          output+="\">Proxima pagina</a>"
          output+="<html>"
          return output
      @expose()
      def form(self):
          output="<html>"
          output+="<body><form action=\"/\" method=\"post\" enctype=\"multipart/form-data\" id=\"filepic\">"
          output+="  Imagem:<br>"
          #output+="  <input type=\"text\" name=\"imagem\"><br>"
          output+="  <input type=\"file\" name=\"file1\"><br>"
          output+="  <input type=\"submit\" value=\"Submit\">"
          output+= " </form> "

          return output
      @expose()
      def conteiner(self, numero=None):
          DBSession.add(Conteiner(numero=numero or ''))
          DBSession.commit()
          return "OK"

config = AppConfig(minimal=True, root_controller=RootController())

config.renderers = ['kajiki']
config.serve_static = True
config.paths['static_files'] = 'img'
config['use_sqlalchemy'] = True
config['sqlalchemy.url'] = 'sqlite:///devdata.db'
from tg.util import Bunch
from sqlalchemy.orm import scoped_session, sessionmaker

DBSession = scoped_session(sessionmaker(autoflush=True, autocommit=False))

def init_model(engine):
    DBSession.configure(bind=engine)
    DeclarativeBase.metadata.create_all(engine)  # Create tables if they do not exist

config['model'] = Bunch(
    DBSession=DBSession,
    init_model=init_model
)

from sqlalchemy.ext.declarative import declarative_base

DeclarativeBase = declarative_base()

from sqlalchemy import Column, Integer, DateTime, String
from datetime import datetime


class Conteiner(DeclarativeBase):
    __tablename__ = 'conteineres'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    numero = Column(String(50), nullable=False)


application = config.make_wsgi_app()

print ("Serving on port 8080...")
httpd = make_server('', 8080, config.make_wsgi_app())
httpd.serve_forever()
