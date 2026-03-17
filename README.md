# Clinica Assistant

Backend en Python para una clinica que recibe mensajes por webhook de Chatwoot, enruta semanticamente con `vLLM`, orquesta con `LangGraph` y mantiene continuidad conversacional con `mem0`.

## Componentes

- `FastAPI` para el webhook `POST`.
- `LangGraph` para el flujo conversacional.
- `vLLM` como backend de inferencia OpenAI-compatible.
- `mem0` para memoria de usuario/conversacion.
- Configuracion local estatica para servicios, horarios, doctores y politicas.

## Setup local

1. Crear y activar entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -e ".[dev]"
```

3. Preparar variables de entorno:

```bash
cp .env.example .env
```

4. Ajustar `config/clinic.json` con los datos reales de la clinica.

5. Levantar `vLLM` localmente. Ejemplo:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8001 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

6. Ejecutar la API:

```bash
uvicorn app.main:create_app --factory --reload
```

## Flujo

1. Chatwoot envia un `POST` al webhook.
2. La API responde inmediatamente con un acuse.
3. En segundo plano se arma el contexto con mensaje, memoria y config clinica.
4. LangGraph decide entre conversacion general o intencion de cita.
5. La respuesta se envia por la API de Chatwoot si esta habilitada; si no, queda registrada en logs.

## Git

El repositorio se inicializa localmente, pero no se hace commit automatico ni se versiona nada por defecto.
