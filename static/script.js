//Javascript scripts 
//Restricts user input into textbox to specified number of words 
const textarea = document.getElementById("user_input") 
//Maximum word count 
const word_Count = 5;

textarea.addEventListener("input", () => {
    const words = textarea.value.trim().split(/\s+/); 
    console.log(textarea.value)
    //Limits user input to less than the specified word count 
    if(words.length > word_Count) { 
        textarea.value = words.slice(0, word_Count).join(' ');
    } 
}) 
