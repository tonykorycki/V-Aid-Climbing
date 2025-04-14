#include <SoftwareSerial.h>
#include <SD.h>

SoftwareSerial mySerial(10, 11);

const int chipSelect = 4;
const int delayTime = 1000;

File footholdFile;

void setup() {
  mySerial.begin(9600);
  Serial.begin(9600);
  Serial.println("Scanning Now");

  if (!SD.begin(chipSelect)) {
    Serial.println("SD card initialization failed!");
    return;
  }

  footholdFile = SD.open("foothold.txt");
  if (!footholdFile) {
    Serial.println("Error opening foothold.txt");
    return;
  }
}

void loop() {
  if (footholdFile && footholdFile.available()) {
    String line = footholdFile.readStringUntil('\n');
    line.trim();

    int commaIndex = line.indexOf(',');
    if (commaIndex > 0 && commaIndex < line.length() - 1) {
      int x = line.substring(0, commaIndex).toInt();
      int y = line.substring(commaIndex + 1).toInt();

      sendGe(x, y);
      delay(delayTime);
    }
  } else {
    footholdFile.close();
    while (true);
  }
}

void sendGe(int x, int y) {
  mySerial.print("G1 X");
  mySerial.print(x);
  mySerial.print(" Y");
  mySerial.println(y);

  Serial.print("The sent X coord through Gcode is: ");
  Serial.print(x);
  Serial.print(" Y");
  Serial.println(y);
}

int main() {
}
