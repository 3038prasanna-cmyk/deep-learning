import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load pre-trained FaceNet model
model = load_model('facenet_keras.h5')
l2_normalizer = Normalizer('l2')

# Function to preprocess face
def preprocess_face(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return np.expand_dims(img, axis=0)

# Load stored face embeddings and names
with open('face_embeddings.pkl', 'rb') as f:
    known_embeddings, known_names = pickle.load(f)

# Initialize face detector once outside loop
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_pp = preprocess_face(face)
        embedding = model.predict(face_pp)[0]
        embedding = l2_normalizer.transform([embedding])[0]

        name = "Unknown"
        max_sim = 0

        # Compare with known embeddings
        for db_emb, db_name in zip(known_embeddings, known_names):
            sim = cosine_similarity([embedding], [db_emb])[0][0]
            if sim > 0.7 and sim > max_sim:
                name = db_name
                max_sim = sim

        # Draw bounding box and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('Face Recognition - Press Q to Exit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
