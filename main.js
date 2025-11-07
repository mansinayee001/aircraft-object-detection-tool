function showLoader(event){
    event.preventDefault(); // stops immediate page refresh 
    document.getElementById("loader").style.display = "block"; // displays loader animation
    setTimeout (function(){
        event.target.submit();
    }, 500); // submits the form after 0.5 seconds 
}

function hideLoader(){
    document.getElementById("loader").style.display = "none";
}