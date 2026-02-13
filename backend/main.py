from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'Deepfake Biometric Auth System Running'}
