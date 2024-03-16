import React, { Component } from "react";
import { Button, Typography, Chip } from "@material-ui/core";

export default class HomePage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedFile: null,
      imageUrl: "",
      tags: [],
      returnImageUrl: "",
    };
    this.handleFileChange = this.handleFileChange.bind(this);
    this.handleUpload = this.handleUpload.bind(this);
  }

  handleFileChange(event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        this.setState({
          selectedFile: file,
          imageUrl: reader.result,
        });
      };
      reader.readAsDataURL(file);
    }
  }

  async handleUpload() {
    const formData = new FormData();
    formData.append("image", this.state.selectedFile);

    try {
      const requestOptions = {
        method: "POST",
        body: formData,
      };
      const response = await fetch("/api/upload_image", requestOptions);

      if (response.ok) {
        const data = await response.json();

        this.setState({
          tags: data.tags,
          returnImageUrl: data.image,
        });
      } else {
        console.error("Error uploading image:", response.statusText);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  }

  render() {
    const { imageUrl, tags, returnImageUrl } = this.state;

    return (
      <div className="container">
        <div class="header">
          <h1>
            <span>Image Classification</span>
          </h1>
        </div>
        <div className="d-flex flex-column align-items-center">
          <div className="row d-flex align-items-center  justify-content-center">
            {imageUrl && (
              <div className={returnImageUrl ? "card col-5" : "card col-12"}>
                <img src={imageUrl} className="card-image-top image"></img>
              </div>
            )}
            {returnImageUrl && (
              <div className={"ms-3 card col-5"}>
                <img src={returnImageUrl} className="card-image-top image"></img>
              </div>
            )}
          </div>

          <h2>Upload an image to generate Image Tags</h2>
          <div class="d-flex mb-3">
            <input type="file" class="form-control" id="inputGroupFile02" onChange={this.handleFileChange} />
          </div>

          <Button variant="contained" color="primary" onClick={this.handleUpload} disabled={!this.state.selectedFile}>
            Upload Image
          </Button>
          {tags.length > 0 && (
            <div>
              <Typography variant="h6" component="h3">
                Tags:
              </Typography>
              {tags.map((tag, index) => (
                <Chip key={index} label={`${tag.tag_name}: ${tag.value_percentage}`} style={{ margin: "5px" }} />
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }
}
