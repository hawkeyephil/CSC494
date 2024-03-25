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

        const sentiment = data;

        //Parses the JSON string
        const parsedSentiment = JSON.parse(sentiment); 

        //Extracts the polarity value as a float
        const lexicon_Polarity = parseFloat(parsedSentiment.lexicon_Polarity).toFixed(2);
        const rounded_Lexicon_Polarity = Math.abs(lexicon_Polarity*100)

        const nbow_Class = parsedSentiment.nbow_Class; 
        const nbow_Probability = parseFloat(parsedSentiment.nbow_Probability).toFixed(2);
        const rounded_nbow_Probability = Math.abs(nbow_Probability*100) 

        const cnn_Class = parsedSentiment.cnn_Class; 
        const cnn_Probability = parseFloat(parsedSentiment.cnn_Probability).toFixed(2);
        const rounded_cnn_Probability = Math.abs(cnn_Probability*100)

        //Gets the lexicon progress bar, icon, and label elements 
        const lexiconSentimentBar = document.getElementById('lexicon-sentiment-bar'); 
        const lexiconSentimentIcon = document.getElementById('lexicon-sentiment-icon'); 
        const lexiconSentimentLabel = document.querySelector('#lexicon-sentiment-label'); 
        //const subjectivityLabel = document.querySelector('#subjectivity-label') 

        //Gets the nbow progress bar, icon, and label elements
        const nbowSentimentBar = document.getElementById('nbow-sentiment-bar') 
        const nbowSentimentIcon = document.getElementById('nbow-sentiment-icon')
        const nbowClassLabel = document.querySelector('#nbow-sentiment-label') 

        //Gets the cnn progress bar, icon, and label elements
        const cnnSentimentBar = document.getElementById('cnn-sentiment-bar') 
        const cnnSentimentIcon = document.getElementById('cnn-sentiment-icon')
        const cnnClassLabel = document.querySelector('#cnn-sentiment-label')

        //Update the width based on the sentiment bars 
        lexiconSentimentBar.style.width = `${rounded_Lexicon_Polarity}%`; 
        nbowSentimentBar.style.width = `${rounded_nbow_Probability}%`; 
        cnnSentimentBar.style.width = `${rounded_cnn_Probability}%`; 

        //Updates the background color and message based on polarity 
        if(lexicon_Polarity > 0.25) { 
          lexiconSentimentBar.style.backgroundColor = 'rgb(140, 193, 82)'; 
          lexiconSentimentBar.innerHTML = `Polarity Score: ${lexicon_Polarity}`; 
          lexiconSentimentIcon.classList.remove('fa-meh'); 
          lexiconSentimentIcon.classList.remove('fa-frown-open')
          lexiconSentimentIcon.classList.add('fa-grin', 'icon-positive'); 
          lexiconSentimentLabel.style.backgroundColor = 'rgb(140, 193, 82)'; 
          //subjectivityLabel.style.backgroundColor = 'rgb(140, 193, 82)';
        } 
        else if(lexicon_Polarity < -0.25) {
          lexiconSentimentBar.style.backgroundColor = 'rgb(193, 82, 82)'; 
          lexiconSentimentBar.innerHTML = `Polarity Score: ${lexicon_Polarity}`; 
          lexiconSentimentIcon.classList.remove('fa-meh'); 
          lexiconSentimentIcon.classList.remove('fa-grin')
          lexiconSentimentIcon.classList.add('fa-frown-open', 'icon-negative'); 
          lexiconSentimentLabel.style.backgroundColor = 'rgb(193, 82, 82)'; 
          //subjectivityLabel.style.backgroundColor = 'rgb(193, 82, 82)';
        } 
        else {
          lexiconSentimentBar.style.backgroundColor = 'rgb(138, 138, 138)'; 
          lexiconSentimentBar.style.color = 'rgb(0,0,0)' 
          lexiconSentimentBar.innerHTML = `Polarity Score: ${lexicon_Polarity}`; 
          lexiconSentimentIcon.classList.remove('fa-frown-open'); 
          lexiconSentimentIcon.classList.remove('fa-grin')
          lexiconSentimentIcon.classList.add('fa-meh', 'icon-neutral'); 
          lexiconSentimentLabel.style.backgroundColor = 'rgb(138, 138, 138)'; 
          //subjectivityLabel.style.backgroundColor = 'rgb(138, 138, 138)';
        }

        if(nbow_Class == 'Positive') { 
          nbowSentimentBar.style.backgroundColor = 'rgb(140, 193, 82)'; 
          nbowSentimentBar.innerHTML = `Prediction Probability: ${rounded_nbow_Probability}%`; 
          nbowSentimentIcon.classList.remove('fa-meh'); 
          nbowSentimentIcon.classList.remove('fa-frown-open')
          nbowSentimentIcon.classList.add('fa-grin', 'icon-positive'); 
          nbowClassLabel.style.backgroundColor = 'rgb(140, 193, 82)'; 
        } 
        else if(nbow_Class == 'Negative') {
          nbowSentimentBar.style.backgroundColor = 'rgb(193, 82, 82)'; 
          nbowSentimentBar.innerHTML = `Prediction Probability: ${rounded_nbow_Probability}%`; 
          nbowSentimentIcon.classList.remove('fa-meh'); 
          nbowSentimentIcon.classList.remove('fa-grin')
          nbowSentimentIcon.classList.add('fa-frown-open', 'icon-negative'); 
          nbowClassLabel.style.backgroundColor = 'rgb(193, 82, 82)'; 
        }
        else {
          nbowSentimentBar.style.backgroundColor = 'rgb(138, 138, 138)'; 
          nbowSentimentBar.style.color = 'rgb(0,0,0)' 
          nbowSentimentBar.innerHTML = `Prediction Probability: ${rounded_nbow_Probability}%`; 
          nbowSentimentIcon.classList.remove('fa-frown-open'); 
          nbowSentimentIcon.classList.remove('fa-grin')
          nbowSentimentIcon.classList.add('fa-meh', 'icon-neutral');
          nbowClassLabel.style.backgroundColor = 'rgb(138, 138, 138)'; 
        } 

        if(cnn_Class == 'Positive') { 
          cnnSentimentBar.style.backgroundColor = 'rgb(140, 193, 82)'; 
          cnnSentimentBar.innerHTML = `Prediction Probability: ${rounded_cnn_Probability}%`; 
          cnnSentimentIcon.classList.remove('fa-meh'); 
          cnnSentimentIcon.classList.remove('fa-frown-open')
          cnnSentimentIcon.classList.add('fa-grin', 'icon-positive'); 
          cnnClassLabel.style.backgroundColor = 'rgb(140, 193, 82)'; 
        } 
        else if(cnn_Class == 'Negative') {
          cnnSentimentBar.style.backgroundColor = 'rgb(193, 82, 82)'; 
          cnnSentimentBar.innerHTML = `Prediction Probability: ${rounded_cnn_Probability}%`; 
          cnnSentimentIcon.classList.remove('fa-meh'); 
          cnnSentimentIcon.classList.remove('fa-grin')
          cnnSentimentIcon.classList.add('fa-frown-open', 'icon-negative'); 
          cnnClassLabel.style.backgroundColor = 'rgb(193, 82, 82)'; 
        }
        else {
          cnnSentimentBar.style.backgroundColor = 'rgb(138, 138, 138)'; 
          cnnSentimentBar.style.color = 'rgb(0,0,0)' 
          cnnSentimentBar.innerHTML = `Prediction Probability: ${rounded_cnn_Probability}%`; 
          cnnSentimentIcon.classList.remove('fa-frown-open'); 
          cnnSentimentIcon.classList.remove('fa-grin')
          cnnSentimentIcon.classList.add('fa-meh', 'icon-neutral');
          cnnClassLabel.style.backgroundColor = 'rgb(138, 138, 138)'; 
        }
    } 
    catch (error) {
        console.error('Error fetching sentiment:', error);
    }
  }
  // Call the function to fetch sentiment data
  getSentiment(); 