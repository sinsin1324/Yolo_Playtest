Manually add yolo weights to weights folder in order to run the program.

Command:
g++ -std=c++17 -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_dnn -lopencv_videoio -lopencv_imgcodecs /Users/sinsin/Desktop/playing_with_vision/src/*.cpp -o /Users/sinsin/Desktop/playing_with_vision/build/a.out
