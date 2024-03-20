//Javascript scripts 
//Restricts user input into textbox to specified number of words 
const textarea = document.getElementById("user_input") 
//Maximum word count 
const word_Count = 500;

textarea.addEventListener("input", () => {
    const words = textarea.value.trim().split(/\s+/); 
    console.log(textarea.value)
    //Limits user input to less than the specified word count 
    if(words.length > word_Count) { 
        textarea.value = words.slice(0, word_Count).join(' ');
    } 
}) 

//Updates the sentiment bar 
async function getSentiment() {
    try {
        const response = await fetch('/polarity_score', {
        method: 'POST',
        });
        const data = await response.text();

        const polarity = data;

        //Parses the JSON string
        const parsedResponse = JSON.parse(polarity);

        //Extracts the polarity value as a float
        const polarityValue = parseFloat(parsedResponse.polarity).toFixed(2); 
        const roundedPolarityValue = Math.abs(polarityValue*100);

        //Gets the progress bar element
        const progressBar = document.getElementById('sentiment-bar'); 
        //Update the width based on the polarity
        progressBar.style.width = `${roundedPolarityValue}%`; 
        //Updates the background color and message based on polarity 
        if(polarityValue > 0.25) { 
          progressBar.style.backgroundColor = `rgb(140, 193, 82)`; 
          progressBar.innerHTML = `Positive: ${polarityValue}`;
        } 
        else if(polarityValue < -0.25) {
          progressBar.style.backgroundColor = 'rgb(193, 82, 82)'; 
          progressBar.innerHTML = `Negative: ${polarityValue}`;
        } 
        else {
          progressBar.style.backgroundColor = 'rgb(138, 138, 138)'; 
          progressBar.style.color = 'rgb(0,0,0)' 
          progressBar.innerHTML = `Neutral: ${polarityValue}`;
        }

    } 
    catch (error) {
        console.error('Error fetching sentiment:', error);
    }
  }
  // Call the function to fetch sentiment data
  getSentiment();