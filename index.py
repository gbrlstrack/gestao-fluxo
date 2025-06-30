# Contador de Pessoas com OpenCV (Versão Simples)
import cv2

# Inicializa o classificador de corpo humano (pré-treinado no OpenCV)
body_cascade = cv2.HOGDescriptor()
body_cascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Captura da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona o frame para acelerar o processamento
    frame = cv2.resize(frame, (640, 480))

    # Converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta pessoas no frame
    boxes, _ = body_cascade.detectMultiScale(gray)

    # Desenha retângulos nas pessoas detectadas
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe a contagem
    cv2.putText(frame, f"Pessoas detectadas: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostra o resultado
    cv2.imshow('Contador de Pessoas', frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
