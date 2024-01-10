function new_transcription_element(speaker_name, speaker_avatar) {
    var avatar_container = document.createElement("div");
    avatar_container.className = "avatar-container";

    var avatar_img = document.createElement("div");
    avatar_img.innerHTML = "<img class='avatar' src='img/" + speaker_avatar + "' \>";

    var avatar_name = document.createElement("div");
    avatar_name.className = "avatar-name";
    avatar_name.innerHTML = speaker_name;

    var dummy_element = document.createElement("div");

    avatar_container.appendChild(avatar_img);
    avatar_container.appendChild(avatar_name);
    avatar_container.appendChild(dummy_element);

    document.getElementById("main-wrapper").appendChild(avatar_container);
}

function new_text_element(text) {
    var text_container = document.createElement("div");
    text_container.className = "text-container";
    text_container.style.maxWidth = "500px";

    var text_element = document.createElement("div");
    text_element.innerHTML = "<p>" + text + "</p>";

    var dummy_element = document.createElement("div");

    text_container.appendChild(text_element);
    text_container.appendChild(dummy_element);

    document.getElementById("main-wrapper").appendChild(text_container);
}

function new_transcription_time_element(time) {
    var text_container = document.createElement("div");
    text_container.className = "transcription-timing-container";
    text_container.style.maxWidth = "500px";

    var text_element = document.createElement("div");
    text_element.innerHTML = "<span>WhisperLive - Transcription time: " + time + "ms</span>";

    var dummy_element = document.createElement("div");

    text_container.appendChild(text_element);
    text_container.appendChild(dummy_element);

    document.getElementById("main-wrapper").appendChild(text_container);
}

function new_llm_time_element(time) {
    var text_container = document.createElement("div");
    text_container.className = "llm-timing-container";
    text_container.style.maxWidth = "500px";

    var first_response_text_element = document.createElement("div");
    first_response_text_element.innerHTML = "<span>Phi-2 first response time: " + time + "ms</span>";

    var complete_response_text_element = document.createElement("div");
    complete_response_text_element.innerHTML = "<span>Phi-2 complete response time: " + time + "ms</span>";

    var dummy_element = document.createElement("div");

    text_container.appendChild(first_response_text_element);
    text_container.appendChild(complete_response_text_element);
    text_container.appendChild(dummy_element);

    document.getElementById("main-wrapper").appendChild(text_container);
}

function new_whisper_speech_time_element(time) {
    var text_container = document.createElement("div");
    text_container.className = "whisperspeech-timing-container";
    text_container.style.maxWidth = "500px";

    var text_element = document.createElement("div");
    text_element.innerHTML = "<span>WhisperSpeech response time: " + time + "ms</span>";

    var dummy_element = document.createElement("div");

    text_container.appendChild(text_element);
    text_container.appendChild(dummy_element);

    document.getElementById("main-wrapper").appendChild(text_container);
}

function WebSocketEntry() {
    var ws = new WebSocket("ws://localhost:9998/collabora");
    ws.onopen = function() {
        ws.send("Message to send");
        console.log("Message is sent...");
    };

    ws.onmessage = function (evt) { 
        var received_msg = evt.data;
        console.log("Message is received...");
    };

    ws.onclose = function() { 
        console.log("Connection is closed..."); 
    };
}

document.addEventListener('DOMContentLoaded', function() {
    new_transcription_element("Marcus", "0.png");
    new_text_element("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.");
    new_transcription_time_element("35");

    new_transcription_element("Phi-2", "Phi.svg");
    new_text_element("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.");
    new_llm_time_element("35");
    new_whisper_speech_time_element("200");

    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
 }, false);
