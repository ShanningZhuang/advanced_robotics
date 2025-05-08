document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prev');
    const nextBtn = document.getElementById('next');
    const slideCounter = document.getElementById('slide-counter');
    
    let currentSlide = 0;
    const totalSlides = slides.length;
    
    // Update slide counter
    const updateCounter = () => {
        slideCounter.textContent = `${currentSlide + 1}/${totalSlides}`;
    };
    
    // Show current slide
    const showSlide = () => {
        slides.forEach((slide, index) => {
            slide.classList.remove('active');
            slide.style.transform = `translateX(${(index - currentSlide) * 100}%)`;
        });
        slides[currentSlide].classList.add('active');
        updateCounter();
    };
    
    // Navigation functions
    const goToNextSlide = () => {
        if (currentSlide < totalSlides - 1) {
            currentSlide++;
            showSlide();
        }
    };
    
    const goToPrevSlide = () => {
        if (currentSlide > 0) {
            currentSlide--;
            showSlide();
        }
    };
    
    // Event listeners
    nextBtn.addEventListener('click', goToNextSlide);
    prevBtn.addEventListener('click', goToPrevSlide);
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'Space') {
            goToNextSlide();
        } else if (e.key === 'ArrowLeft') {
            goToPrevSlide();
        }
    });
    
    // Swipe navigation for touch devices
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
    
    const handleSwipe = () => {
        const swipeThreshold = 50;
        if (touchEndX < touchStartX - swipeThreshold) {
            // Swipe left - next slide
            goToNextSlide();
        } else if (touchEndX > touchStartX + swipeThreshold) {
            // Swipe right - previous slide
            goToPrevSlide();
        }
    };
    
    // Initialize
    showSlide();
}); 