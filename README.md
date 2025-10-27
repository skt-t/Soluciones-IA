# Soluciones-IA
Contexto:
Es un sistema de consultas para conocer informacion sobre piercings, el codigo se ejecuta modulo por modulo.

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

En el espacio virtual instalaremos las dependencias (en el txt "requerimientos" se puede encontrar otra forma de instalar las dependencias): 

pip install crewai langchain-openai python-dotenv duckduckgo-search wikipedia


# Proyecto de Agente Experto: Asistente de Piercings

## 1. Descripción del Proyecto

Este proyecto implementa un agente de IA funcional utilizando el framework **CrewAI**. El agente actúa como un "Asistente Experto en Modificación Corporal", especializado en piercings. Es capaz de mantener una conversación, buscar información en la web y en Wikipedia, y escribir informes en archivos locales.

El objetivo es cumplir con los requisitos de la Evaluación Parcial N°2 (ISY0101), demostrando la integración de herramientas, memoria, y planificación en un agente funcional.

---

## 2. Arquitectura y Diseño (IL2.4)

### Justificación del Diseño

Se optó por una arquitectura de **Agente Único ("Experto")** en lugar de un crew multi-agente. Esta decisión de diseño fue estratégica por dos motivos principales:

1.  **Optimización de API:** Un agente único es mucho más eficiente en el consumo de llamadas a la API de OpenAI. Dado que el plan gratuito tiene un `Rate Limit` estricto (ej. 50 llamadas/día), un diseño multi-agente (que puede gastar 4-5 llamadas por consulta) agotaría la cuota instantáneamente. El diseño de agente único garantiza la funcionalidad completa sin exceder los límites.
2.  **Cumplimiento de la Rúbrica:** Un solo agente puede cumplir con todos los indicadores de logro (IL) si se diseña correctamente. El `goal` del agente se ha diseñado para forzar el **razonamiento (IL2.1)** y la **planificación (IL2.3)**, mientras que la **memoria (IL2.2)** se habilita a nivel de Agente y Crew.

### Componentes Principales

* **Framework:** `CrewAI` (sobre `LangChain`)
* **LLM:** `gpt-4o` (a través de `langchain-openai`)
* **Agente:** `piercing_expert_agent` (Agente único con múltiples herramientas)
* **Herramientas:**
    * `DuckDuckGoTool` (Consulta Web)
    * `WikipediaTool` (Consulta Fáctica)
    * `file_write_tool` (Escritura de Archivos)

### Diagrama de Orquestación (IE9)

El flujo de trabajo es gestionado por el Agente Experto, que decide qué herramienta utilizar basándose en la consulta del usuario.

```mermaid
graph TD
    A[Usuario] -->|Consulta: {query}| B(Crew: piercing_crew);
    B --> C{Agente Experto (Razonamiento)};
    C -->|IL2.2| D[Memoria (Historial)];
    C -->|IL2.1| E[Herramienta: DuckDuckGo];
    C -->|IL2.1| F[Herramienta: Wikipedia];
    C -->|IL2.1| G[Herramienta: Escritor de Archivos];
    D --> C;
    E --> H[Respuesta];
    F --> H;
    G --> H;
    H --> A;