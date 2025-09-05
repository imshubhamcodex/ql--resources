#include <MD_MAX72xx.h>
#include <SPI.h>

#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES 4
#define DATA_PIN D7
#define CS_PIN D4
#define CLK_PIN D5

MD_MAX72XX leds = MD_MAX72XX(HARDWARE_TYPE, DATA_PIN, CLK_PIN, CS_PIN, MAX_DEVICES);

// Grid size
#define ROWS 2
#define COLS 8
#define ACTIONS 4  // up, down, left, right

bool obstacle[ROWS][COLS] = {
  { false, true, false, false, false, false, false, false },
  { false, false, false, true, false, false, false, false }
};

// Q-table
float Q[ROWS][COLS][ACTIONS];

// Counts
int episodeNumber = 0;
int trajectoryLength = 0;

// Params
float alpha = 0.5;
float discount = 0.95;
float epsilon = 0.1;

// Agent state
int row = 0, col = 0;
int goalRow = 1, goalCol = 7;

// Training control
bool trainingPaused = false;
bool trainingDone = false;

// Serial input state machine
enum InputState { NONE, WAIT_ROW, WAIT_COL };
InputState inputState = NONE;
int tempRow = -1;

// ---------------- Utility Functions ----------------
void clearMatrix() { leds.clear(); }

void drawAgent(int r, int c) {
  clearMatrix();
  int startY = r * 3 + 1;
  int startX = c * 4 + 1;
  for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++)
      leds.setPoint(startY + dy, startX + dx, true);
  leds.update();
}

void animateMove(int oldR, int oldC, int newR, int newC, int action) {
  clearMatrix();
  int baseY = oldR * 3 + 1;
  int baseX = oldC * 4 + 1;

  switch (action) {
    case 0: for (int dy = -1; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) leds.setPoint(baseY + dy, baseX + dx, true); break; // UP
    case 1: for (int dy = 0; dy < 3; dy++) for (int dx = 0; dx < 2; dx++) leds.setPoint(baseY + dy, baseX + dx, true); break;  // DOWN
    case 2: for (int dx = -1; dx < 2; dx++) for (int dy = 0; dy < 2; dy++) leds.setPoint(baseY + dy, baseX + dx, true); break;  // LEFT
    case 3: for (int dx = 0; dx < 3; dx++) for (int dy = 0; dy < 2; dy++) leds.setPoint(baseY + dy, baseX + dx, true); break;  // RIGHT
  }

  leds.update();
  delay(50);
  drawAgent(newR, newC);
}

// ---------------- Q-Learning ----------------
int chooseAction(int r, int c) {
  if ((float)random(100) / 100.0 < epsilon) return random(ACTIONS);
  float bestQ = -9999;
  int bestA = 0;
  for (int a = 0; a < ACTIONS; a++) if (Q[r][c][a] > bestQ) { bestQ = Q[r][c][a]; bestA = a; }
  return bestA;
}

int reward(int r, int c) {
  if (r == goalRow && c == goalCol) return 10;
  if (obstacle[r][c]) return -5;
  return -1;
}

void moveAgent(int action) {
  int newR = row, newC = col;
  if (action == 0 && row > 0) newR--;
  if (action == 1 && row < ROWS - 1) newR++;
  if (action == 2 && col > 0) newC--;
  if (action == 3 && col < COLS - 1) newC++;

  int rwd = reward(newR, newC);
  float maxNext = -9999;
  for (int a = 0; a < ACTIONS; a++) if (Q[newR][newC][a] > maxNext) maxNext = Q[newR][newC][a];

  Q[row][col][action] += alpha * (rwd + discount * maxNext - Q[row][col][action]);
  animateMove(row, col, newR, newC, action);

  row = newR;
  col = newC;
  trajectoryLength++;

  if (row == goalRow && col == goalCol) {
    episodeNumber++;
    Serial.print("Episode ");
    Serial.print(episodeNumber);
    Serial.print(" finished! Trajectory length: ");
    Serial.println(trajectoryLength);
    trajectoryLength = 0;

    delay(200);
    do {
      row = random(ROWS);
      col = random(COLS);
    } while (obstacle[row][col] || (row == goalRow && col == goalCol));
  }
}

// ---------------- Optimal Path ----------------
void followOptimalPathAnimate(int startR, int startC) {
  int r = startR, c = startC;
  struct Cell { int row, col; };
  Cell path[ROWS * COLS];
  int pathLen = 0;

  clearMatrix();
  drawAgent(r, c);

  while (!(r == goalRow && c == goalCol)) {
    path[pathLen++] = {r, c};
    float bestQ = -9999;
    int bestA = -1;
    for (int a = 0; a < ACTIONS; a++) if (Q[r][c][a] > bestQ) { bestQ = Q[r][c][a]; bestA = a; }
    if (bestA == -1) bestA = 3;

    int newR = r, newC = c;
    switch (bestA) {
      case 0: if (r > 0) newR--; break;
      case 1: if (r < ROWS - 1) newR++; break;
      case 2: if (c > 0) newC--; break;
      case 3: if (c < COLS - 1) newC++; break;
    }

    if (newR != r || newC != c) animateMove(r, c, newR, newC, bestA);
    r = newR; c = newC;
  }
  path[pathLen++] = {goalRow, goalCol};

  // Highlight full path
  for (int i = 0; i < pathLen; i++) {
    int y = path[i].row * 3 + 1;
    int x = path[i].col * 4 + 1;
    for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) leds.setPoint(y + dy, x + dx, true);
  }
  leds.update();
}

// ---------------- Serial Input ----------------
void handleSerialInput() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.equalsIgnoreCase("p")) {
      trainingPaused = true;
      inputState = NONE;
      Serial.println("Training paused!");
    } else if (input.equalsIgnoreCase("r")) {
      if (trainingPaused) { trainingPaused = false; inputState = NONE; Serial.println("Training resumed!"); }
    } else if (trainingPaused || trainingDone) {
      // Non-blocking step-by-step input
      if (inputState == NONE) { Serial.println("Enter row number (0-1):"); inputState = WAIT_ROW; }
      else if (inputState == WAIT_ROW) {
        tempRow = input.toInt();
        if (tempRow >= 0 && tempRow < ROWS) { inputState = WAIT_COL; Serial.println("Enter column number (0-7):"); }
        else Serial.println("Invalid row! Enter 0-1:");
      } else if (inputState == WAIT_COL) {
        int tempCol = input.toInt();
        if (tempCol >= 0 && tempCol < COLS) {
          Serial.print("Running optimal path from: "); Serial.print(tempRow); Serial.print(", "); Serial.println(tempCol);
          followOptimalPathAnimate(tempRow, tempCol);
          inputState = NONE;
        } else Serial.println("Invalid column! Enter 0-7:");
      }
    }
  }
}

// ---------------- Setup & Loop ----------------
void setup() {
  leds.begin();
  Serial.begin(9600);
  clearMatrix();
  randomSeed(analogRead(A0));

  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      for (int a = 0; a < ACTIONS; a++)
        Q[r][c][a] = 0.0;
}

void loop() {
  handleSerialInput();

  if (!trainingDone && !trainingPaused) {
    moveAgent(chooseAction(row, col));
    delay(40);

    if (episodeNumber > 100) {
      trainingDone = true;
      Serial.println("Training done!");
      Serial.println("You can now enter row and column to run optimal path.");
    }
  }
}
