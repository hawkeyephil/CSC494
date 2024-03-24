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

        //Gets the progress bar, icon, and label elements
        const sentimentBar = document.getElementById('sentiment-bar'); 
        const sentimentIcon = document.getElementById('sentiment-icon'); 
        const polarityLabel = document.querySelector('#polarity-label'); 
        const subjectivityLabel = document.querySelector('#subjectivity-label') 

        //Update the width based on the polarity
        sentimentBar.style.width = `${roundedPolarityValue}%`; 
        //Updates the background color and message based on polarity 
        if(polarityValue > 0.25) { 
          sentimentBar.style.backgroundColor = 'rgb(140, 193, 82)'; 
          sentimentBar.innerHTML = `Positive: ${polarityValue}`; 
          sentimentIcon.classList.remove('fa-meh'); 
          sentimentIcon.classList.remove('fa-frown-open')
          sentimentIcon.classList.add('fa-grin', 'icon-positive'); 
          polarityLabel.style.backgroundColor = 'rgb(140, 193, 82)'; 
          subjectivityLabel.style.backgroundColor = 'rgb(140, 193, 82)';
        } 
        else if(polarityValue < -0.25) {
          sentimentBar.style.backgroundColor = 'rgb(193, 82, 82)'; 
          sentimentBar.innerHTML = `Negative: ${polarityValue}`; 
          sentimentIcon.classList.remove('fa-meh'); 
          sentimentIcon.classList.remove('fa-grin')
          sentimentIcon.classList.add('fa-frown-open', 'icon-negative'); 
          polarityLabel.style.backgroundColor = 'rgb(193, 82, 82)'; 
          subjectivityLabel.style.backgroundColor = 'rgb(193, 82, 82)';
        } 
        else {
          sentimentBar.style.backgroundColor = 'rgb(138, 138, 138)'; 
          sentimentBar.style.color = 'rgb(0,0,0)' 
          sentimentBar.innerHTML = `Neutral: ${polarityValue}`; 
          sentimentIcon.classList.remove('fa-frown-open'); 
          sentimentIcon.classList.remove('fa-grin')
          sentimentIcon.classList.add('fa-meh', 'icon-neutral'); 
          polarityLabel.style.backgroundColor = 'rgb(138, 138, 138)'; 
          subjectivityLabel.style.backgroundColor = 'rgb(138, 138, 138)';
        }

    } 
    catch (error) {
        console.error('Error fetching sentiment:', error);
    }
  }
  // Call the function to fetch sentiment data
  getSentiment(); 