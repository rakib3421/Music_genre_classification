document.addEventListener('DOMContentLoaded', function () {
    // Get the file input and uploaded audio elements
    const fileInput = document.getElementById('audioUpload');
    const uploadedAudio = document.getElementById('uploadedAudio');

    // Handle file input change event
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];  // Get the selected file

        if (file) {
            const reader = new FileReader();  // Create a FileReader instance
            reader.onload = function(e) {
                uploadedAudio.src = e.target.result;  // Set the src of the audio to the file data URL
                uploadedAudio.style.display = 'block';  // Show the uploaded audio
            };
            reader.readAsDataURL(file);  // Read the file as a data URL
        }
    });

    // You can add more JavaScript code here if needed
});