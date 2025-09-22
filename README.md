# Soluciones-IA
Contexto:
Level-Up Gamer es una tienda online chilena especializada en productos para gamers (consolas, computadores, accesorios, merchandising y servicio técnico). Opera solo en línea, despachando a todo Chile. Su visión es consolidarse como la tienda gamer líder del país, con un programa de fidelización y comunidad.

problematica: Level - Up no cuenta con el personal suficiente para responder preguntas frecuentes y necesita responder a dudas de nuevos clientes.

Nosotros propusimos usar una ia que reponda a sus preguntas basandose en casos que ya han solucionado, generando una respuesta automatica con la solucion al problema del cliente.

# ANTES DE INSTALAR

Asegurarse de tener los tokens de GitHub y OpenAi como el ejemplo que esta en el archivo: ".env.example"
Estamos usando gpt-4o asegurarse de utilizar el modelo correcto, si usted esta usando otro modelo, modificarlo en model="gpt-4o" en la linea 241.

# Instalacion

Antes de iniciarlo opcionalmente creamos un espacio virtual (Recomendado):

- MAC: python -m venv venv 
- Windows: python -m venv venv

Entraremos al espacio virtual con 
- MAC: source .venv/bin/activate
- Windows: .\.venv\Scripts\Activate.ps1

(Abajo instrucciones de como instalar las dependencias, estando o no en un espacio virtual)

Para probarlo usaremos streamlit, primero hay que instalar las dependencias.
Para ello usar el documento de texto "REQUERIMIENTOS.txt" 
usando "pip" o "!pip" si esta en mac, abajo dos ejemplos:

- MAC: !pip install -r REQUERIMIENTOS.txt
- Windows: pip install -r REQUERIMIENTOS.txt

Para iniciarlo copiar: streamlit run evaluacion.py

