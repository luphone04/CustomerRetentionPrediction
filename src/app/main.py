from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict

app = FastAPI(
    title="Customer Retention Intelligence System",
    description="An API for predicting customer churn using a pre-trained machine learning model.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}

class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


def gradio_predict(
        gender, Partner, Dependents, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges)
    }
    result = predict(data)

    if "Likely to churn" in result:
        return f"{result}", "high-risk"
    else:
        return f"{result}", "low-risk"


with gr.Blocks(theme=gr.themes.Soft(), title="Churn Predictor") as demo:
    gr.Markdown("# Customer Churn Predictor")
    gr.Markdown("Predict if a customer will churn based on their profile.")

    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Demographics")
                    gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                    Partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
                    Dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")

                    gr.Markdown("### Phone")
                    PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                    MultipleLines = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")

                with gr.Column():
                    gr.Markdown("### Internet Services")
                    InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                    OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                    OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                    DeviceProtection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                    TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                    StreamingTV = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No")
                    StreamingMovies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Contract & Billing")
                    Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                    PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
                    PaymentMethod = gr.Dropdown([
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"
                    ], label="Payment Method", value="Electronic check")

                with gr.Column():
                    gr.Markdown("### Charges")
                    tenure = gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100)
                    MonthlyCharges = gr.Number(label="Monthly Charges ($)", value=70.0, minimum=0, maximum=200)
                    TotalCharges = gr.Number(label="Total Charges ($)", value=70.0, minimum=0, maximum=10000)

        # Right column - Output (sticky)
        with gr.Column(scale=1):
            gr.Markdown("### Prediction Result")
            output = gr.Textbox(label="Result", lines=3, show_label=False, container=True)
            predict_btn = gr.Button("Predict Churn", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### Quick Examples")

            high_risk_btn = gr.Button("High Risk Customer", size="sm")
            low_risk_btn = gr.Button("Low Risk Customer", size="sm")

    # Button actions
    predict_btn.click(
        fn=gradio_predict,
        inputs=[gender, Partner, Dependents, PhoneService, MultipleLines,
                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                TechSupport, StreamingTV, StreamingMovies, Contract,
                PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges],
        outputs=[output]
    )

    # High risk example
    def set_high_risk():
        return ("Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
                "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 1, 85.0, 85.0)

    high_risk_btn.click(
        fn=set_high_risk,
        outputs=[gender, Partner, Dependents, PhoneService, MultipleLines,
                 InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                 TechSupport, StreamingTV, StreamingMovies, Contract,
                 PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges]
    )

    # Low risk example
    def set_low_risk():
        return ("Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
                "Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 60, 45.0, 2700.0)

    low_risk_btn.click(
        fn=set_low_risk,
        outputs=[gender, Partner, Dependents, PhoneService, MultipleLines,
                 InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                 TechSupport, StreamingTV, StreamingMovies, Contract,
                 PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges]
    )


app = gr.mount_gradio_app(app, demo, path="/ui")