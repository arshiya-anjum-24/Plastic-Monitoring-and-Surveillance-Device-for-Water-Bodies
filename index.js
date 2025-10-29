const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const mongoose = require("mongoose");
const path = require("path");
const { Gpio } = require("pigpio");

// --- CHECK pigpio connection ---
try {
    const test = new Gpio(4, { mode: Gpio.INPUT });
    console.log("✅ pigpio daemon connected.");
} catch (err) {
    console.error("❌ pigpio connection failed. Run: sudo systemctl start pigpiod");
    process.exit(1);
}

// --- MOTOR SETUP ---
const motorL_fwd = new Gpio(17, { mode: Gpio.OUTPUT });
const motorL_rev = new Gpio(27, { mode: Gpio.OUTPUT });
const motorR_fwd = new Gpio(22, { mode: Gpio.OUTPUT });
const motorR_rev = new Gpio(23, { mode: Gpio.OUTPUT });

// --- SMOOTH TRANSITION SETTINGS ---
const TRANSITION_STEPS = 10;
const STEP_DELAY_MS = 30;

let currentL = 0;
let currentR = 0;

function setMotor(pinFwd, pinRev, speed) {
    if (speed > 0) {
        pinFwd.pwmWrite(speed);
        pinRev.digitalWrite(0);
    } else if (speed < 0) {
        pinFwd.digitalWrite(0);
        pinRev.pwmWrite(Math.abs(speed));
    } else {
        pinFwd.digitalWrite(0);
        pinRev.digitalWrite(0);
    }
}

function transitionTo(targetL, targetR) {
    let leftFlip = Math.sign(currentL) !== Math.sign(targetL);
    let rightFlip = Math.sign(currentR) !== Math.sign(targetR);

    if (leftFlip || rightFlip) {
        // Stage 1: ramp to 0 for direction change
        const steps1 = TRANSITION_STEPS / 2;
        const leftStep1 = leftFlip ? -currentL / steps1 : (targetL - currentL) / TRANSITION_STEPS;
        const rightStep1 = rightFlip ? -currentR / steps1 : (targetR - currentR) / TRANSITION_STEPS;
        let count1 = 0;

        const interval1 = setInterval(() => {
            count1++;
            currentL += leftStep1;
            currentR += rightStep1;
            setMotor(motorL_fwd, motorL_rev, Math.round(currentL));
            setMotor(motorR_fwd, motorR_rev, Math.round(currentR));

            if (count1 >= steps1) {
                clearInterval(interval1);
                // Stage 2: ramp to target
                const steps2 = TRANSITION_STEPS / 2;
                const leftStep2 = leftFlip ? targetL / steps2 : 0;
                const rightStep2 = rightFlip ? targetR / steps2 : 0;
                let count2 = 0;

                const interval2 = setInterval(() => {
                    count2++;
                    currentL += leftStep2;
                    currentR += rightStep2;
                    setMotor(motorL_fwd, motorL_rev, Math.round(currentL));
                    setMotor(motorR_fwd, motorR_rev, Math.round(currentR));

                    if (count2 >= steps2) clearInterval(interval2);
                }, STEP_DELAY_MS);
            }
        }, STEP_DELAY_MS);
    } else {
        // Smooth same-direction ramp
        const leftStep = (targetL - currentL) / TRANSITION_STEPS;
        const rightStep = (targetR - currentR) / TRANSITION_STEPS;
        let count = 0;

        const interval = setInterval(() => {
            count++;
            currentL += leftStep;
            currentR += rightStep;
            setMotor(motorL_fwd, motorL_rev, Math.round(currentL));
            setMotor(motorR_fwd, motorR_rev, Math.round(currentR));

            if (count >= TRANSITION_STEPS) clearInterval(interval);
        }, STEP_DELAY_MS);
    }
}

// --- MOVEMENT COMMANDS ---
function moveForward() { transitionTo(100, -20); }
function moveLeft() { transitionTo(85, 70); }
function moveRight() { transitionTo(70, -70); }
function moveBackward() { transitionTo(-50, -50); }
function stop() { transitionTo(0, 0); }

// --- MONGODB SETUP ---
mongoose.connect("mongodb://localhost:27017/boatLogs")
    .then(() => console.log("✅ MongoDB connected."))
    .catch(err => console.error("❌ MongoDB error:", err.message));

const Detection = mongoose.model("Detection", {
    lat: Number,
    lon: Number,
    confidence: Number,
    timestamp: Date
});

// --- EXPRESS SERVER ---
const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

let boatPower = false;

// --- API ROUTES ---
app.get("/api/state", (req, res) => res.json({ power: boatPower }));

app.get("/control/:cmd", async (req, res) => {
    const cmd = req.params.cmd.toLowerCase();
    console.log(`⚙️ Command received: ${cmd}`);

    if (cmd === "start") {
        await Detection.deleteMany({});
        boatPower = true;
        console.log("🚤 Boat POWER ON");
        return res.json({ message: "Boat ready" });
    }

    if (cmd === "stop") {
        boatPower = false;
        stop();
        console.log("🛑 Boat STOPPED");
        return res.json({ message: "Boat stopped" });
    }

    if (!boatPower) return res.json({ message: "Start the boat first" });

    switch (cmd) {
        case "forward":
            moveForward();
            break;
        case "left":
            moveLeft();
            break;
        case "right":
            moveRight();
            break;
        case "backward":
            moveBackward();
            break;
        case "halt":
            stop();
            break;
        default:
            return res.status(400).json({ message: "Invalid command" });
    }

    return res.json({ message: `Boat moving ${cmd}` });
});

app.post("/api/detection", async (req, res) => {
    const { lat, lon, confidence } = req.body;
    if (lat && lon && confidence) {
        const det = new Detection({
            lat: parseFloat(lat),
            lon: parseFloat(lon),
            confidence: parseFloat(confidence),
            timestamp: new Date()
        });
        await det.save();
        io.emit("new_detection", det.toObject());
        return res.status(201).json({ message: "Detection logged" });
    }
    res.status(400).json({ message: "Missing data" });
});

app.get("/logs", async (req, res) => {
    const logs = await Detection.find().sort({ timestamp: -1 });
    res.json(logs);
});

// --- SERVER START ---
server.listen(3000, () => console.log("✅ Control server running at http://localhost:3000"));

// --- SAFE SHUTDOWN ---
process.on('SIGINT', () => {
    console.log("\n🧭 Shutting down...");
    stop();
    console.log("⚓ Motors stopped safely. Exiting.");
    process.exit(0);
});
