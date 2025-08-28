document.addEventListener("DOMContentLoaded", () => {
  // Initialize AOS animations
  AOS.init({
    duration: 1000,
    easing: 'ease-out-cubic',
    once: false,
  });

  // Leaflet Map Setup
  const map = L.map("map").setView([20.5937, 78.9629], 5); // Default India

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);

  let marker;

  // Fetch full address using reverse geocoding
  function updateLocation(lat, lng) {
    document.getElementById("latitude").value = lat;
    document.getElementById("longitude").value = lng;

    fetch(`https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lng}`)
      .then(res => res.json())
      .then(data => {
        if (data && data.display_name) {
          const addressInput = document.getElementById("address-display");
       if (addressInput) {
             addressInput.value = data.display_name;
            }


          const addressDisplay = document.getElementById("address-display");
          if (addressDisplay) {
            addressDisplay.value = data.display_name;
          }
        }
      })
      .catch(err => console.error("Address fetch error:", err));
  }

  // Request geolocation permission
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(position => {
      const { latitude, longitude } = position.coords;
      map.setView([latitude, longitude], 13);

      // Add draggable marker
      marker = L.marker([latitude, longitude], { draggable: true }).addTo(map);
      updateLocation(latitude, longitude);

      // Update location when marker is moved
      marker.on("dragend", () => {
        const pos = marker.getLatLng();
        updateLocation(pos.lat, pos.lng);
      });

    }, error => {
      console.warn("Geolocation not allowed or unavailable:", error);
    });
  } else {
    console.warn("Geolocation is not supported by this browser.");
  }
});
