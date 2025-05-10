const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors()); // Enable CORS
app.use(bodyParser.json());

// Mock AI response (replace with your actual model later)
function generateResponse(userInput) {
  const responses = [
    "Your pet might have a mild allergy. Try switching their food.",
    "This could be a sign of an infection. Please consult a vet.",
    "Many pets experience this. Monitor for 24 hours and check for changes.",
  ];
  return responses[Math.floor(Math.random() * responses.length)];
}

// Chat endpoint
app.post('/api/chat', (req, res) => {
  const { message } = req.body;
  console.log("User asked:", message);

  // Simulate processing delay (1-2 seconds)
  setTimeout(() => {
    const response = generateResponse(message);
    res.json({ response });
  }, 1000);
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});