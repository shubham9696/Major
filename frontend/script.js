		let base64Image;
		$("#inputGroupFile01").change(function(){
			let reader = new FileReader();
			reader.onload = function(e){
				let dataURL = reader.result;
				$("#selected-image").attr("src",dataURL);
				base64Image = dataURL.replace("data:image/jpeg;base64,","");
			}
			var filename = $(this).val();
			filename = filename.substring(filename.lastIndexOf('\\') + 1, filename.length);
			$(this).next('.custom-file-label').html(filename);
			reader.readAsDataURL($("#inputGroupFile01")[0].files[0]);
		});

		$("#predict-button").click(function(event){
			let message={
				image: base64Image
			}

			console.log("hello");

			$.post("http://localhost:5000/predict",JSON.stringify(message),function(response){

				var images = document.getElementById('img-container');
				while(images.firstChild){
    				images.removeChild(images.firstChild);
				}

			    console.log(response.predictions);

			    response.predictions.forEach(function(obj) {

			      var img = document.createElement("img");
			      img.setAttribute("src", obj[0]);
				  img.setAttribute("class", "img-responsive img-thumbnail");

                  var brand = document.createElement("h6");
                  brand.innerHTML = obj[2]["brand"];
                  brand.setAttribute("class", "brand-class");

                  var name = document.createElement("p");
                  name.innerHTML = obj[2]["name"];
                  name.setAttribute("class", "text-class");

                  var price = document.createElement("span");
                  price.innerHTML = obj[2]["price"];
                  price.setAttribute("class", "brand-class");

                  var mrp = document.createElement("span");
                  var s = document.createElement("s");
                  s.innerHTML = obj[2]["mrp"];
                  mrp.appendChild(s);
                  mrp.setAttribute("class", "mrp-class");

                  var discount = document.createElement("span");
                  discount.innerHTML = obj[2]["discount"];
                  discount.setAttribute("class", "discount-class")

			      var aTag = document.createElement('a');
				  aTag.setAttribute('href',obj[1]);
				  aTag.setAttribute("class", "imgBox")
				  aTag.appendChild(img);
				  aTag.appendChild(brand);
				  aTag.appendChild(name);
				  aTag.appendChild(price);
				  aTag.appendChild(mrp);
				  aTag.appendChild(discount);

				  images.appendChild(aTag);
				});
			});

		});