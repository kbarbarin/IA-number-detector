// Configuration du canvas
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let session = null;

// Paramètres de dessin
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000000';

// Initialisation du canvas
ctx.fillStyle = '#FFFFFF';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Événements de dessin pour souris
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Événements de dessin pour tactile
canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', stopDrawing);

// Boutons
document.getElementById('clearBtn').addEventListener('click', clearCanvas);
document.getElementById('predictBtn').addEventListener('click', predict);

// Fonctions de dessin
function startDrawing(e) {
    isDrawing = true;
    ctx.beginPath();
    const rect = canvas.getBoundingClientRect();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouchStart(e) {
    e.preventDefault();
    isDrawing = true;
    ctx.beginPath();
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
}

function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
    ctx.stroke();
}

function clearCanvas() {
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetPrediction();
}

function resetPrediction() {
    document.getElementById('predictedDigit').textContent = '-';
    document.getElementById('confidence').textContent = '-';
    document.getElementById('probabilityBars').innerHTML = '';
}

// Charger le modèle ONNX avec données externes
async function loadModel() {
    try {
        document.getElementById('loading').style.display = 'block';
        console.log('🔄 Chargement du modèle ONNX...');
        
        // Étape 1: Charger le fichier .data
        console.log('📥 Chargement des données externes...');
        const dataResponse = await fetch('model.onnx.data');
        if (!dataResponse.ok) {
            throw new Error('Impossible de charger model.onnx.data');
        }
        const dataBuffer = await dataResponse.arrayBuffer();
        console.log(`✅ Données chargées: ${(dataBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);
        
        // Étape 2: Charger le modèle ONNX
        console.log('📥 Chargement du modèle ONNX...');
        const modelResponse = await fetch('model.onnx');
        if (!modelResponse.ok) {
            throw new Error('Impossible de charger model.onnx');
        }
        const modelBuffer = await modelResponse.arrayBuffer();
        console.log(`✅ Modèle chargé: ${(modelBuffer.byteLength / 1024).toFixed(2)} KB`);
        
        // Étape 3: Créer la session avec les données externes
        console.log('🔧 Création de la session ONNX...');
        const options = {
            externalData: [
                {
                    data: new Uint8Array(dataBuffer),
                    path: 'model.onnx.data'
                }
            ]
        };
        
        session = await ort.InferenceSession.create(modelBuffer, options);
        
        console.log('✅ Modèle ONNX chargé avec succès!');
        document.getElementById('loading').style.display = 'none';
        return true;
    } catch (error) {
        console.error('❌ Erreur lors du chargement du modèle:', error);
        console.error('Type d\'erreur:', error.name);
        console.error('Message:', error.message);
        console.error('Stack:', error.stack);
        
        let errorMsg = 'Erreur: Impossible de charger le modèle.\n\n';
        if (error.message.includes('fetch') || error.message.includes('Impossible de charger')) {
            errorMsg += 'Problème de chargement réseau.\n';
            errorMsg += '✓ Vérifiez que le serveur HTTP est lancé:\n';
            errorMsg += '  python3 -m http.server 8000\n';
            errorMsg += '✓ Accédez au site via: http://localhost:8000\n';
            errorMsg += '✓ Les fichiers model.onnx et model.onnx.data doivent exister\n';
        } else {
            errorMsg += 'Détails: ' + error.message;
        }
        
        alert(errorMsg);
        document.getElementById('loading').style.display = 'none';
        return false;
    }
}

// Prétraiter l'image du canvas
function preprocessCanvas() {
    // Créer un canvas temporaire de 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Redimensionner l'image
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Obtenir les données de l'image
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Créer un tableau Float32Array pour ONNX
    const input = new Float32Array(1 * 1 * 28 * 28);
    
    // Normaliser les pixels (grayscale et normalisation MNIST)
    for (let i = 0; i < 28; i++) {
        for (let j = 0; j < 28; j++) {
            const idx = (i * 28 + j) * 4;
            // Convertir en niveau de gris et inverser (MNIST a des chiffres blancs sur fond noir)
            const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            const normalized = (255 - gray) / 255.0; // Inverser
            
            // Normalisation MNIST: mean=0.1307, std=0.3081
            input[i * 28 + j] = (normalized - 0.1307) / 0.3081;
        }
    }
    
    return input;
}

// Fonction softmax pour calculer les probabilités
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b);
    return scores.map(s => s / sum);
}

// Prédire le chiffre
async function predict() {
    // Charger le modèle si ce n'est pas déjà fait
    if (!session) {
        const loaded = await loadModel();
        if (!loaded) return;
    }
    
    try {
        // Prétraiter l'image
        const inputData = preprocessCanvas();
        
        // Créer le tensor d'entrée
        const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
        
        // Exécuter l'inférence
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        
        // Obtenir les résultats
        const output = results.output.data;
        
        // Calculer les probabilités avec softmax
        const probabilities = softmax(Array.from(output));
        
        // Trouver la classe prédite
        const predictedClass = probabilities.indexOf(Math.max(...probabilities));
        const confidence = probabilities[predictedClass];
        
        // Afficher les résultats
        displayPrediction(predictedClass, confidence, probabilities);
        
    } catch (error) {
        console.error('❌ Erreur lors de la prédiction:', error);
        alert('Erreur lors de la prédiction. Vérifiez la console pour plus de détails.');
    }
}

// Afficher les résultats de la prédiction
function displayPrediction(digit, confidence, probabilities) {
    // Afficher le chiffre prédit
    document.getElementById('predictedDigit').textContent = digit;
    
    // Afficher la confiance
    document.getElementById('confidence').textContent = `${(confidence * 100).toFixed(1)}%`;
    
    // Afficher les barres de probabilité
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';
    
    for (let i = 0; i < 10; i++) {
        const probability = probabilities[i];
        const percentage = (probability * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        const label = document.createElement('span');
        label.className = 'digit-label';
        label.textContent = i;
        
        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        
        const fill = document.createElement('div');
        fill.className = 'probability-fill';
        fill.style.width = `${probability * 100}%`;
        fill.textContent = percentage > 5 ? `${percentage}%` : '';
        
        bar.appendChild(fill);
        item.appendChild(label);
        item.appendChild(bar);
        probabilityBars.appendChild(item);
    }
}

// Charger le modèle au démarrage
console.log('🚀 Initialisation de l\'application...');
loadModel();

