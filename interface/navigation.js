function setupNavigation() {
    const menu = document.getElementById("mobile-menu");
    const openButton = document.getElementById("mobile-menu-open");
    const closeButton = document.getElementById("mobile-menu-close");
    const desktopNavigation = document.querySelector(".site-header nav");
    const mobileNavigation = document.getElementById("mobile-navigation");

    if (!menu || !openButton || !closeButton || !desktopNavigation
        || !mobileNavigation || typeof menu.showModal !== "function") {
        return;
    }

    const phoneLayout = matchMedia("(max-width: 850px)");

    for (const link of desktopNavigation.querySelectorAll("a")) {
        mobileNavigation.append(link.cloneNode(true));
    }

    function closeMenu() {
        if (menu.open) menu.close();
    }

    openButton.addEventListener("click", () => {
        if (!phoneLayout.matches || menu.open) return;
        menu.showModal();
        openButton.setAttribute("aria-expanded", "true");
        document.body.classList.add("menu-open");
    });

    closeButton.addEventListener("click", closeMenu);

    menu.addEventListener("close", () => {
        openButton.setAttribute("aria-expanded", "false");
        document.body.classList.remove("menu-open");
        if (phoneLayout.matches) {
            openButton.focus({ preventScroll: true });
        }
    });

    menu.addEventListener("click", event => {
        if (event.target !== menu) return;
        const box = menu.getBoundingClientRect();
        const outside = event.clientX < box.left || event.clientX > box.right
            || event.clientY < box.top || event.clientY > box.bottom;
        if (outside) closeMenu();
    });

    mobileNavigation.addEventListener("click", event => {
        if (event.target.closest("a")) closeMenu();
    });

    phoneLayout.addEventListener("change", () => {
        if (!phoneLayout.matches) closeMenu();
    });

    window.addEventListener("pageshow", () => {
        closeMenu();
        openButton.setAttribute("aria-expanded", "false");
        document.body.classList.remove("menu-open");
    });

    document.body.classList.add("navigation-ready");
}

setupNavigation();
