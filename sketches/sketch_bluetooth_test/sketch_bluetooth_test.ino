/*  ___   ___  ___  _   _  ___   ___   ____ ___  ____  
 * / _ \ /___)/ _ \| | | |/ _ \ / _ \ / ___) _ \|    \ 
 *| |_| |___ | |_| | |_| | |_| | |_| ( (__| |_| | | | |
 * \___/(___/ \___/ \__  |\___/ \___(_)____)___/|_|_|_|
 *                  (____/ 
 * Prueba del bluetooth.
 * @author blackzafiro
 * PIN: 1234
 */

#define MAX_PACKETSIZE 32    //Serial receive buffer

char buffUART[MAX_PACKETSIZE];
unsigned int buffUARTIndex = 0;
unsigned long preUARTTick = 0;

/* Inicialización */
void setup() {
  Serial.begin(9600);  // Bluetooth baud rate, only 9600
  Serial2.begin(9600);
  Serial.println(" .-~*´¨¯¨`*·~-. Bluetooth on .-~*´¨¯¨`*·~-.");
}

/* En cada ciclo escucha por el bluetooth. */
void loop() {
  doUARTTick();
}

/**
 * UART significa receptor/transmisor asíncrono universal
 * define un protocolo para intercambiar datos en serie entre dos dispositivos,
 * es el protocolo utilizado por el módulo Bluetooth HC-02 del Osoyoo.
 */
void doUARTTick() {
  // Basado en el tutorial de OSOYOO Bluetooth
  char dato_UART = 0;
  if(Serial2.available()) {
    size_t len = Serial2.available();
    Serial.print("Info came in... ");
    uint8_t sbuf[len + 1];
    sbuf[len] = 0x00;
    Serial2.readBytes(sbuf, len);
    Serial.print("info was read... ");
    memcpy(buffUART + buffUARTIndex, sbuf, len);//ensure that the serial port can read the entire frame of data
    buffUARTIndex += len;
    preUARTTick = millis();
    if(buffUARTIndex >= MAX_PACKETSIZE - 1) {
      buffUARTIndex = MAX_PACKETSIZE - 2;
      preUARTTick = preUARTTick - 200;
    }
  }

  //APP send flag to modify the obstacle avoidance parameters
  if(buffUARTIndex > 0 && (millis() - preUARTTick >= 100)) {
    //data ready
    buffUART[buffUARTIndex] = 0x00;
    dato_UART = buffUART[0];
    buffUARTIndex = 0;
  }

  if (dato_UART) {
    Serial.print("UART dato: ");
    Serial.println(dato_UART);
    Serial2.print(1);
  }
}