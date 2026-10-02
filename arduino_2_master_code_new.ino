#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include "Communication.h"

static Communication Comm(25); // Arduino 2 Address is 25

// Pin Definitions
#define RPWM 6
#define LPWM 5
#define R_EN 4
#define L_EN 7
#define SOLENOID_PIN 3
#define HOME_SENSOR A1
#define SD_CS 9

int runID = 0;
String Characteristic = "ati";

void extendActuator() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 255);
}

void retractActuator() {
  analogWrite(RPWM, 255);
  analogWrite(LPWM, 0);
}

void stopActuator() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
}

void setup() {
  Serial.begin(115200);

  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);

  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);

  pinMode(HOME_SENSOR, INPUT_PULLUP);
  pinMode(SOLENOID_PIN, OUTPUT);

  // ADDED: Required for reliable SPI communication on Arduino Uno.
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);

  // ADDED: Configure the SD card chip-select pin explicitly.
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);

  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);
  digitalWrite(SOLENOID_PIN, LOW);

  Serial.println(F("--- ARDUINO 2 BOOT ---"));

  if (!SD.begin(SD_CS)) {
    Serial.println(F("SD Init Failed!"));

    // If SD fails, stay on RED as a warning.
    Comm.Transmit(10, "SET_RED");

    // ADDED: Do not start runs if the SD card is unavailable.
    return;
  } else {
    Serial.println(F("SD Card Ready."));
  }

  delay(2000);

  // --- MAIN LOOP: RUN 5 TIMES ---
  for (int i = 0; i < 5; i++) {
    Serial.print(F("\n>>> STARTING RUN "));
    Serial.print(i + 1);
    Serial.println(F(" of 5 <<<"));

    executeOneRun();

    delay(2000);
  }

  // --- FINISHED: SET GREEN LIGHT ---
  Comm.Transmit(10, "SET_GREEN");

  Serial.println(F("\n************************************"));
  Serial.println(F("ALL 5 RUNS COMPLETE."));
  Serial.println(F("TRAFFIC LIGHT IS GREEN. SAFE TO EJECT."));
  Serial.println(F("************************************"));
}

void moveHome(int &counter) {
  Serial.println(F("Homing Actuator..."));

  retractActuator();

  while (digitalRead(HOME_SENSOR) == HIGH);

  stopActuator();

  counter++;

  delay(500);
}

void executeOneRun() {
  int homeCounter = 0;

  runID++;

  // 1. Tell Arduino 1 to go RED and move Home.
  Comm.Transmit(10, "SET_RED");

  moveHome(homeCounter);

  // 2. Extend until ToF trigger.
  Serial.println(F("Step: Extending..."));

  Comm.Transmit(10, "WAIT_TRIGGER");

  extendActuator();

  bool triggered = false;

  while (!triggered) {
    Comm.Transmit(10, "CHECK_TRIGGER");

    // Listen for the "YES" stop command.
    unsigned long listenStart = millis();

    while (millis() - listenStart < 50) {
      if (Comm.Received() == STRING_MESSAGE) {
        String msg = Comm.GetStringMessage();

        if (msg == "YES") {
          stopActuator();

          triggered = true;

          Serial.println(F("Stop Signal Received."));

          break;
        }
      }
    }

    delay(5);
  }

  // ==========================================
  // 3. Fire Solenoid (Hit-Hold Mode) & Retract
  // ==========================================

  Serial.println(F("Step: Firing Solenoid (Hit)..."));

  // HIT PHASE: Full power.
  analogWrite(SOLENOID_PIN, 255);

  delay(100);

  // HOLD PHASE: Approximately 90% power.
  Serial.println(F("Step: Dropping to Hold Power..."));

  analogWrite(SOLENOID_PIN, 230);

  // Actuator moves back while the solenoid remains active.
  moveHome(homeCounter);

  // TURN OFF: Completely switch off the solenoid.
  analogWrite(SOLENOID_PIN, 0);

  // ADDED: Explicitly force the control pin LOW.
  digitalWrite(SOLENOID_PIN, LOW);

  Serial.println(F("Step: Solenoid Deactivated."));

  // ADDED: Allow electrical noise and supply voltage to settle.
  delay(500);

  // ==========================================
  // 4. Data collection and SD writing
  // ==========================================

  Serial.println(F("Step: Writing to SD..."));

  char fileName[13];

  sprintf(
    fileName,
    "%s_R%d.CSV",
    Characteristic.c_str(),
    runID
  );

  Serial.print(F("Creating file: "));
  Serial.println(fileName);

  // ADDED: Restore SD communication after solenoid operation.
  if (!SD.begin(SD_CS)) {
    Serial.println(F("SD Error: Reinitialization failed!"));

    Comm.Transmit(10, "SET_RED");

    return;
  }

  File dataFile = SD.open(fileName, FILE_WRITE);

  if (dataFile) {
    // Header for the CSV.
    dataFile.println(
      F("Sample,M0,M1,M2,M3,M4,M5,M6,M7,M8,M9,ToF")
    );

    int sampleCount = 0;

    unsigned long startTime = millis();

    // Collect 100 samples or stop after 10 seconds.
    while (
      sampleCount < 100 &&
      millis() - startTime < 10000
    ) {
      Comm.Transmit(10, "GET_DATA");

      String fullRow = "";

      int partsReceived = 0;

      unsigned long timeout = millis();

      // Wait for both parts of the split data message.
      while (
        millis() - timeout < 600 &&
        partsReceived < 2
      ) {
        if (Comm.Received() == STRING_MESSAGE) {
          fullRow += Comm.GetStringMessage();

          partsReceived++;
        }
      }

      if (partsReceived == 2) {
        dataFile.print(sampleCount);

        dataFile.print(F(","));

        dataFile.println(fullRow);

        sampleCount++;
      }

      delay(20);
    }

    dataFile.flush();
    dataFile.close();

    Serial.print(F("File Saved Successfully: "));
    Serial.println(fileName);

  } else {
    Serial.println(F("SD Error: Could not open file for writing!"));
  }
}

void loop() {
  // Empty: logic runs once in setup() for exactly five runs.
}