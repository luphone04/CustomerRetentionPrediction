from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict

