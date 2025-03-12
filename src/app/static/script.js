document.addEventListener("DOMContentLoaded", function () {
  const chatForm = document.getElementById("chat-gen");
  const userInput = document.querySelector("input[name='user_input']");
  const responseBox = document.querySelector(".chat-answer");
  
  chatForm.onsubmit = async function (event) {
      event.preventDefault();
      
      // Append new responses to keep history
      const userMessage = `<p><strong>You:</strong> ${userInput.value}</p>`;
      const doormanMessage = `<p><strong>Doorman:</strong> <span id="response-text"></span></p>`;
      responseBox.innerHTML += userMessage + doormanMessage;
      const responseText = document.getElementById("response-text");

      const formData = new FormData(chatForm);

      try {
          const response = await fetch("/llm", {
              method: "POST",
              body: formData
          });

          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          while (true) {
              const { value, done } = await reader.read();
              if (done) break;
              responseText.innerHTML += decoder.decode(value, { stream: true });  // Append streamed text
          }

          // Reload page and clean input
            userInput.value = '';
            location.reload();

          // Play audio after text is fully loaded
          const audio = document.querySelector('audio');
          if (audio) {
            audio.play().catch(error => console.warn("Auto-play blocked:", error));
          }

      } catch (error) {
          responseText.innerHTML = `<span style="color:red;">Error: ${error.message}</span>`;
      }
  };
});

// Play audio when window loads
window.addEventListener("load", function () {
  const audio = document.querySelector('audio');
  if (audio) {
    audio.play().catch(error => console.warn("Auto-play blocked:", error));
  }
});
