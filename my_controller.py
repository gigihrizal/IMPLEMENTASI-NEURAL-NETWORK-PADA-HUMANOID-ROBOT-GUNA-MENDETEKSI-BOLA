from controller import Robot, Camera
import cv2
import numpy as np

# Inisialisasi Webots
robot = Robot()
camera = robot.getDevice("CameraTop")
camera.enable(10)  # Update citra setiap 10ms

# inisialisasi motor untuk sendi kaki kiri dan kanan
left_motor = robot.getDevice("LHipPitch")
right_motor = robot.getDevice("RHipPitch")
anklekiri_motor = robot.getDevice("LAnklePitch")
anklekanan_motor = robot.getDevice("RAnklePitch")
lengankiri_motor = robot.getDevice("LShoulderPitch")
lengankanan_motor = robot.getDevice("RShoulderPitch")
dengkulkiri_motor = robot.getDevice("LKneePitch")
dengkulkanan_motor = robot.getDevice("RKneePitch")
kepala_motor = robot.getDevice("HeadPitch")

# atur posisi awal motor
left_motor.setPosition(0.0)
right_motor.setPosition(0.0)
anklekiri_motor.setPosition(0.0)
anklekanan_motor.setPosition(0.0)
lengankiri_motor.setPosition(0.0)
lengankanan_motor.setPosition(0.0)
kepala_motor.setPosition(0.2)

# Inisialisasi Hard Cascade Classifier
ball_cascade = cv2.CascadeClassifier("ball_cascade.xml")  # Ganti dengan path ke file XML yang sesuai
#human_cascade = cv2.CascadeClassifier("Human.xml")

# Menggabungkan cascade
#cascade = cv2.CascadeClassifier()

# Memuat file cascade individu
#cascade.load("ball_cascade.xml")
#cascade.load("Human.xml")

# Fungsi untuk mendeteksi bola
def detect_ball(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    balls = ball_cascade.detectMultiScale(gray, 1.1, 4)
    return balls

# Loop utama
while robot.step(10) != -1:
    # Mendapatkan citra dari kamera
    image = camera.getImage()
    image = np.frombuffer(image, np.uint8)
    image = np.reshape(image, (camera.getHeight(), camera.getWidth(), 4))
    image = image[:, :, :3]  # Menghapus saluran alpha
    image = cv2.UMat(image)  # Mengubah tipe data citra menjadi cv2.UMat

    # Mendeteksi bola
    balls = detect_ball(image)

    # Menandai bola dengan persegi pada citra
    for (x, y, w, h) in balls:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Mengonversi citra UMat kembali ke tipe data numpy array
    image = cv2.UMat.get(image)

    # Menampilkan citra hasil deteksi
    cv2.imshow("Ball Detection", image)
    cv2.waitKey(1)
    
    # gerakan dengkul kanan
    #dengkulkanan_motor.setPosition(-0.09)
    #dengkulkiri_motor.setPosition(0.0)
    #robot.step(500)
  
    # gerakan dengkul kiri
    #dengkulkiri_motor.setPosition(0.0)
    #dengkulkanan_motor.setPosition(-0.09)
    #robot.step(500)
    
    # gerakkan kaki kiri maju
    #right_motor.setPosition(0.0)
    #robot.step(250)
    #left_motor.setPosition(0.25) 
    #robot.step(500)
      
    # gerakkan kaki kanan maju
    #left_motor.setPosition(0.0)
    #right_motor.setPosition(0.25)
    #robot.step(500)
   
cv2.destroyAllWindows()