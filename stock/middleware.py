class XFrameOptionsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if '/django_plotly_dash/' in request.path:
            response['X-Frame-Options'] = 'SAMEORIGIN'

        return response