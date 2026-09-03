import os
import time
import warnings
from dotenv import load_dotenv
from langgraph.types import Command
from email_agent import app

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

def run_deployment_demo():
    print("==================================================")
    print("  DEMO: CONSUMO DE DESPLIEGUE & DOUBLE-TEXTING")
    print("==================================================")
    
    thread_id = "thread_deploy_101"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. Envío inicial de correo urgente
    initial_email = {
        "email_content": "¡Me cobraron dos veces la suscripción! Es urgente un reembolso.",
        "sender_email": "cliente@empresa.com",
        "email_id": "MSG_001"
    }
    
    print("\n[1] Enviando correo al agente despechado...")
    result = app.invoke(initial_email, config)
    
    # 2. Manejo de Pausa (Interrupt / Human-in-the-loop)
    if "__interrupt__" in result:
        details = result["__interrupt__"][0].value
        print(f"\n⚠️  GRAFO PAUSADO POR INTERRUPT:")
        print(f"    - ID Correo: {details['email_id']}")
        print(f"    - Acción requerida: {details['action']}")
        print(f"    - Borrador generado: {details['draft'][:80]}...")
        
        # Simulación de aprobación por parte del supervisor
        print("\n[2] Operador humano aprueba el envío...")
        final_res = app.invoke(
            Command(resume={"approved": True}),
            config=config
        )
        print("✅ Estado de cierre:", "Procesado con éxito.")

    # 3. Demostración de Estrategia Double-Texting
    print("\n--------------------------------------------------")
    print("[3] SIMULACIÓN DE DOUBLE-TEXTING")
    print("--------------------------------------------------")
    print("Escenario: El usuario envía un segundo mensaje mientras el primero procesa.")
    
    # Configuración de multithreading / multicontext
    # En LangGraph Cloud / Platform, el parámetro 'multitask_strategy' controla el double-texting:
    # Options: 'reject' (rechaza el 2do), 'enqueue' (pone en cola) o 'interrupt' (cancela el 1ro).
    
    print("Estrategia configurada en Servidor: 'enqueue' (cola ordenada por thread_id)")
    
    msg_1 = {"email_content": "Consulta 1: ¿Tienen descuento de estudiante?", "sender_email": "a@test.com", "email_id": "M1"}
    msg_2 = {"email_content": "Consulta 2: Olviden lo anterior, prefiero el plan anual.", "sender_email": "a@test.com", "email_id": "M2"}
    
    cfg_dt = {"configurable": {"thread_id": "thread_double_text"}}
    
    print("-> Enviando Mensaje 1...")
    res_dt1 = app.invoke(msg_1, cfg_dt)
    print("-> Enviando Mensaje 2 en el mismo hilo...")
    res_dt2 = app.invoke(msg_2, cfg_dt)
    
    print("Ambos mensajes fueron procesados en orden en el hilo 'thread_double_text'.")

if __name__ == "__main__":
    run_deployment_demo()