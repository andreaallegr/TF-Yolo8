# Descrizione progetto
## TIPO TASK: Object Recognition/Detection
## OBBIETTIVO: Partendo dal modello YOLO8 addestrare un nuovo modello tramite  transfer learning per riconoscere oggetti/persone custom 
## DETTAGLI:

- creazione di un dataset utile (con definizione di bbox)
- come fare data augmentation
- training e finetuning  su GPU con il pesi preadddestrati small e large
- prestazioni e accuracy su immagini statiche e stream video (o webcam)

Il notebook da cui partirei potrebbe essere questo
https://keras.io/examples/vision/yolov8/

proverei un primo approccio con questo dataset...
https://universe.roboflow.com/achala/indoor-objects-3gtbo
