const video = document.getElementById('video');
const canvas = document.createElement('canvas');
document.body.appendChild(canvas);

async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
        video.srcObject = stream;
    } catch (err) {
        console.error("An error occurred: " + err);
    }
}

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('./models')
]).then(startVideo);

video.addEventListener('play', async () => {
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video)
            .withFaceLandmarks()
            .withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        const labeledFaceDescriptors = await loadLabeledImage();
        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

        resizedDetections.forEach(detection => {
            const box = detection.detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: faceMatcher.findBestMatch(detection.descriptor).toString() });
            drawBox.draw(canvas);
        });
    }, 100);
});

async function loadLabeledImage() {
    const labels = ['박신원', '백대현', '천예준', '최우진'];
    return Promise.all(
        labels.map(async label => {
            const description = [];
            const img = await faceapi.fetchImage('known/' + label + '.jpg');
            const detections = await faceapi.detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor();
            description.push(detections.descriptor);
            return new faceapi.LabeledFaceDescriptors(label, description);
        })
    );
}
