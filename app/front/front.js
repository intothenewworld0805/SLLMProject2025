function talk() {
    const question = $("#txtMsg").val();

    $.ajax({
        type: "POST",
        url: "http://127.0.0.1:8000/ask",
        data: JSON.stringify({ question: question }),
        contentType: "application/json",
        success: function (res) {
            $("#txtOut").val(res.answer);
        },
        error: function (xhr, status, error) {
            console.error("에러 발생:", error);
        }
    });
}
