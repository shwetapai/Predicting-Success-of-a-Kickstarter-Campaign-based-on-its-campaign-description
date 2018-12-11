$(function(){

    $('button').click(function(){
      init_display();
		  $('#result-analyzing').show();
    	var kickstarterUrl = $('#hyperlink').val();
    	$.ajax({
    		url: '/predict',
    		data: $('form').serialize(),
    		type: 'POST',
			  dataType: "json",
    		success: function(response){
				  analyze(response.curr_prob);
    		},
    		error: function(error){
    			showError();
    		}
    	});
    });
});

function init_display() {
  $('#result-best').hide();
  $('#result-better').hide();
  $('#result-good').hide();
  $('#result-error').hide();
  $('#result-new-prob').hide();
}

function analyze(curr_prob,new_prob) {

  $('#result-analyzing').hide();
	if(curr_prob >= 66) {
		$('#btn-result-best').html(curr_prob);
		$('#result-best').show();
	} else if(curr_prob < 66 && curr_prob >= 33) {
		$('#btn-result-better').html(curr_prob);
		$('#result-better').show();
	} else {
		$('#btn-result-good').html(curr_prob);
		$('#result-good').show();
	}
  $('#btn-new-prob').html(new_prob);
  $('#result-new-prob').show();
}

function showError() {
  $('#result-analyzing').hide();
	$('#result-error').html("<strong>Error!</strong> A problem occurred while analyzing your link.");
	$('#result-error').show();
}
