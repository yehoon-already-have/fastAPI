import React, { useState } from "react";
import axios from "axios";

function ImageUploader() {
  const [predictions, setPredictions] = useState([]);
  const [imageSrc, setImageSrc] = useState(null);

  const uploadImage = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    
    console.log("이미지 업로드 시작");
    const response = await axios.post("http://localhost:8000/predict", formData);
    
    setPredictions(response.data.predictions);
    setImageSrc(response.data.image);
  };

  return (
    <div>
      <input type="file" onChange={uploadImage} />
      
      {imageSrc && <img src={imageSrc} alt="Uploaded" style={{ maxWidth: "300px", marginTop: "10px" }} />}
      
      {predictions.length > 0 && (
        <div>
          <h3>Predictions:</h3>
          <ul>
            {predictions.map((pred, index) => (
              <li key={index}>
                {pred.class} - {pred.probability * 100}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;