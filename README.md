# Hand Detector
Это простой в использовании класс для распознавания рук и пальцев.
![Example image](https://raw.githubusercontent.com/prostoiChelovek/handDetector/master/example.png)

## Как пользоваться?
1. Подключить библиотеку: `#include "handDetector.hpp"`
2. Создать экземпляр класса: `HandDetector hd;`
3. Удалить фон(не обязательно, но улучшает качество): `hd.deleteBg(frame, bg, img);`
4. Найти руки:
   * `hd.detectHands_range(imgHSV, lower, upper);`
   * 
   
       
       hd.loadCascade("path/to/cascade.xml");
       hd.detectHands_Cascade(img);
       

5. Найти пальцы: `hd.getFingers();`
6. Найти центр рук: `hd.getCenters();`
7. Найти спмые высокие кончики пальцев: `hd.getHigherFingerstips();`
8. Нарисовать руки: `hd.drawHands(frame, Scalar(255, 0, 100), 2);`

## WARNING
Этот класс я писал для себя и под свои нуужды. Также я имею мало опыта в работе с С++ и OpenCV.
Любая критика приветсвуется! 